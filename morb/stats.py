from morb.base import Stats
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

import theano

rng = RandomStreams.seed(100)

def gibbs_step(rbm, vmap, units_list, mean_field_for_stats=[], mean_field_for_gibbs=[]):
    # implements a single gibbs step, and makes sure mean field is only used where it should be.
    # returns two vmaps, one for stats and one for gibbs.
    # also enforces consistency between samples, between the gibbs vmap and the stats vmap.
    # the provided lists and vmaps are expected to be COMPLETE. Otherwise, the behaviour is unspecified.
    
    # first, find out which units we need to sample for the stats vmap, and which for the gibbs vmap.
    # Mean field will be used for the others.
    units_sample_stats = units_list[:] # make a copy
    units_mean_field_stats = []
    for u in mean_field_for_stats:
        if u in units_sample_stats:
            units_sample_stats.remove(u) # remove all mean field units from the sample list
            units_mean_field_stats.append(u) # add them to the mean field list instead

    units_sample_gibbs = units_list[:]
    units_mean_field_gibbs = []
    for u in mean_field_for_gibbs:
        if u in units_sample_gibbs:
            units_sample_gibbs.remove(u) # remove all mean field units from the sample list
            units_mean_field_gibbs.append(u) # add them to the mean field list instead

    # now we can compute the total list of units to sample.
    # By sampling them all in one go, we can enforce consistency.
    units_sample = list(set(units_sample_gibbs + units_sample_stats))
    sample_vmap = rbm.sample(units_sample, vmap)
    units_mean_field = list(set(units_mean_field_gibbs + units_mean_field_stats))
    mean_field_vmap = rbm.mean_field(units_mean_field, vmap)
    
    # now, construct the gibbs and stats vmaps
    stats_vmap = dict((u, sample_vmap[u]) for u in units_sample_stats)
    stats_vmap.update(dict((u, mean_field_vmap[u]) for u in units_mean_field_stats))
    gibbs_vmap = dict((u, sample_vmap[u]) for u in units_sample_gibbs)
    gibbs_vmap.update(dict((u, mean_field_vmap[u]) for u in units_mean_field_gibbs))
        
    return stats_vmap, gibbs_vmap


def cd_stats(rbm, v0_vmap, visible_units, hidden_units, context_units=[], k=1, mean_field_for_stats=[], mean_field_for_gibbs=[], persistent_vmap=None):
    # mean_field_for_gibbs is a list of units for which 'mean_field' should be used during gibbs sampling, rather than 'sample'.
    # mean_field_for_stats is a list of units for which 'mean_field' should be used to compute statistics, rather than 'sample'.

    # complete units lists
    visible_units = rbm.complete_units_list(visible_units)
    hidden_units = rbm.complete_units_list(hidden_units)
    context_units = rbm.complete_units_list(context_units)
    
    # complete the supplied vmap
    v0_vmap = rbm.complete_vmap(v0_vmap)
    
    # extract the context vmap, because we will need to merge it into all other vmaps
    context_vmap = dict((u, v0_vmap[u]) for u in context_units)

    h0_activation_vmap = dict((h, h.activation(v0_vmap)) for h in hidden_units)
    h0_stats_vmap, h0_gibbs_vmap = gibbs_step(rbm, v0_vmap, hidden_units, mean_field_for_stats, mean_field_for_gibbs)
            
    # add context
    h0_activation_vmap.update(context_vmap)
    h0_gibbs_vmap.update(context_vmap)
    h0_stats_vmap.update(context_vmap)
    
    exp_input = [v0_vmap[u] for u in visible_units]
    exp_context = [v0_vmap[u] for u in context_units]
    exp_latent = [h0_gibbs_vmap[u] for u in hidden_units]
    
    # scan requires a function that returns theano expressions, so we cannot pass vmaps in or out. annoying.
    def gibbs_hvh(*args):
        h0_gibbs_vmap = dict(zip(hidden_units, args))
        
        v1_in_vmap = h0_gibbs_vmap.copy()
        v1_in_vmap.update(context_vmap) # add context
        
        v1_activation_vmap = dict((v, v.activation(v1_in_vmap)) for v in visible_units)
        v1_stats_vmap, v1_gibbs_vmap = gibbs_step(rbm, v1_in_vmap, visible_units, mean_field_for_stats, mean_field_for_gibbs)

        h1_in_vmap = v1_gibbs_vmap.copy()
        h1_in_vmap.update(context_vmap) # add context

        h1_activation_vmap = dict((h, h.activation(h1_in_vmap)) for h in hidden_units)
        h1_stats_vmap, h1_gibbs_vmap = gibbs_step(rbm, h1_in_vmap, hidden_units, mean_field_for_stats, mean_field_for_gibbs)
            
        # get the v1 values in a fixed order
        v1_activation_values = [v1_activation_vmap[u] for u in visible_units]
        v1_gibbs_values = [v1_gibbs_vmap[u] for u in visible_units]
        v1_stats_values = [v1_stats_vmap[u] for u in visible_units]
        
        # same for the h1 values
        h1_activation_values = [h1_activation_vmap[u] for u in hidden_units]
        h1_gibbs_values = [h1_gibbs_vmap[u] for u in hidden_units]
        h1_stats_values = [h1_stats_vmap[u] for u in hidden_units]
        
        return v1_activation_values + v1_stats_values + v1_gibbs_values + \
               h1_activation_values + h1_stats_values + h1_gibbs_values
    
    
    # support for persistent CD
    if persistent_vmap is None:
        chain_start = exp_latent
    else:
        chain_start = [persistent_vmap[u] for u in hidden_units]
    
    
    # The 'outputs_info' keyword argument of scan configures how the function outputs are mapped to the inputs.
    # in this case, we want the h1_gibbs_vmap values to map onto the function arguments, so they become
    # h0_gibbs_vmap values in the next iteration. To this end, we construct outputs_info as follows:
    outputs_info = [None] * (len(exp_input)*3) + [None] * (len(exp_latent)*2) + list(chain_start)
    # 'None' indicates that this output is not used in the next iteration.
    
    exp_output_all_list, theano_updates = theano.scan(gibbs_hvh, outputs_info = outputs_info, n_steps = k)
    # we only need the final outcomes, not intermediary values
    exp_output_list = [out[-1] for out in exp_output_all_list]
            
    # reconstruct vmaps from the exp_output_list.
    n_input, n_latent = len(visible_units), len(hidden_units)
    vk_activation_vmap = dict(zip(visible_units, exp_output_list[0:1*n_input]))
    vk_stats_vmap = dict(zip(visible_units, exp_output_list[1*n_input:2*n_input]))
    vk_gibbs_vmap = dict(zip(visible_units, exp_output_list[2*n_input:3*n_input]))
    hk_activation_vmap = dict(zip(hidden_units, exp_output_list[3*n_input:3*n_input+1*n_latent]))
    hk_stats_vmap = dict(zip(hidden_units, exp_output_list[3*n_input+1*n_latent:3*n_input+2*n_latent]))
    hk_gibbs_vmap = dict(zip(hidden_units, exp_output_list[3*n_input+2*n_latent:3*n_input+3*n_latent]))
    
    # add the Theano updates for the persistent CD states:
    if persistent_vmap is not None:
        for u, v in persistent_vmap.items():
            theano_updates[v] = hk_gibbs_vmap[u] # this should be the gibbs vmap, and not the stats vmap!
    
    activation_data_vmap = v0_vmap.copy() # TODO: this doesn't really make sense to have in an activation vmap!
    activation_data_vmap.update(h0_activation_vmap)
    activation_model_vmap = vk_activation_vmap.copy()
    activation_model_vmap.update(context_vmap)
    activation_model_vmap.update(hk_activation_vmap)
    
    stats = Stats(theano_updates) # create a new stats object
    
    # store the computed stats in a dictionary of vmaps.
    stats_data_vmap = v0_vmap.copy()
    stats_data_vmap.update(h0_stats_vmap)
    stats_model_vmap = vk_stats_vmap.copy()
    stats_model_vmap.update(context_vmap)
    stats_model_vmap.update(hk_stats_vmap)
    stats.update({
      'data': stats_data_vmap,
      'model': stats_model_vmap,
    })
            
    stats['data_activation'] = activation_data_vmap
    stats['model_activation'] = activation_model_vmap
        
    return stats

def rescale_activations(activation_vmap, beta):
    """Returns activations after mutiplying them with the values in beta.

    beta should be a two dimensional tensor with shape (1, N) and broadcastable
    in the first dimension.
    """
    return manipulate_vmap(activation_vmap, lambda x: beta * x)

def manipulate_vmap(vmap, f):
    # TODO: check with S
    return dict((v, f(tensor)) for v, tensor in vmap.items())

def pt_stats(rbm, v0_vmap, visible_units, hidden_units, persistent_vmap, beta, k=1, m=1):
    """Returns stats for parallel tempering given a list of inverse temperatures beta.
    
    
    """
    # v0_vmap is the batch of train data
    # k is the number of sampling steps
    # persistent_vmap and v0_vmap should determine the data batch size and the number of chains
    # m is the number of chains that is used for model statistics
    N_chains = persistent_vmap[rbm.v].shape[0]


    # complete units lists
    visible_units = rbm.complete_units_list(visible_units)
    hidden_units = rbm.complete_units_list(hidden_units)
    
    # complete the supplied vmap
    v0_vmap = rbm.complete_vmap(v0_vmap)
    mb_size = theano.tensor.cast(v0_vmap[rbm.v].shape[0], dtype=theano.config.floatX)
    n_chains = theano.tensor.cast(m, dtype=theano.config.floatX)
    
    h0_activation_vmap = dict((h, h.activation(v0_vmap)) for h in hidden_units)

    # compute data dependent gradient component
    h0_stats_vmap = rbm.mean_field_from_activation(h0_activation_vmap)

            
    # scan requires a function that returns theano expressions, so we cannot pass vmaps in or out. annoying.
    # for this reason, the exp lists are used and the identity of the variables is coded in the order.
    exp_input = [v0_vmap[u] for u in visible_units]
    exp_latent = [h0_stats_vmap[u] for u in hidden_units]

    rescale = mb_size / n_chains
    
    def gibbs_hvh(*args):
        # generates a fixed order list to be processed by scan
        h0_gibbs_vmap = dict(zip(hidden_units, args))
        
        # what goes 'in' the rbm to compute the visibles
        v1_in_vmap = h0_gibbs_vmap.copy()
        
        v1_activation_vmap = dict((v, v.activation(v1_in_vmap)) for v in visible_units)
        v1_scaled_activation_vmap = rescale_activations(v1_activation_vmap, beta)
        
        v1_gibbs_vmap_preswap = rbm.sample_from_activation(v1_scaled_activation_vmap)
        # stats contains the values that are used for parameter updating and
        # should only be based on the first m chain(s) that have temperature 1.
        v1_stats_vmap_preswap = manipulate_vmap(v1_gibbs_vmap_preswap, lambda x: x[:m, :])

        h1_in_vmap = v1_gibbs_vmap_preswap.copy()

        h1_activation_vmap = dict((h, h.activation(h1_in_vmap)) for h in hidden_units)

        h1_scaled_activation_vmap = rescale_activations(h1_activation_vmap, beta)
        h1_gibbs_vmap_preswap = rbm.sample_from_activation(h1_scaled_activation_vmap)

        # Select a candidate pair for replica exchange
        rv_i = rng.random_integers((1,), low=0, high=N_chains-2) #the range is inclusive
        rv_u = rng.uniform((1,))
        
        chain_pair = manipulate_vmap(h1_scaled_activation_vmap, lambda x: x[rv_i:rv_i+2, :])
        chain_pair_v = manipulate_vmap(v1_gibbs_vmap_preswap, lambda x: x[rv_i:rv_i+2, :])
        left_chain = manipulate_vmap(h1_scaled_activation_vmap, lambda x: x[rv_i, :])
        right_chain = manipulate_vmap(h1_scaled_activation_vmap, lambda x: x[rv_i+1, :])

        # compute relevant energy scores
        # Note that we could use eigher the energy or the free energy. Using
        # the free energy will sample from p(v) instead of p(v, h) and is what
        # we are interested in.
        # Also note that we don't need to store the h values because the chain
        # always starts from the visible units.
        v_fe_pair = rbm.free_energy_unchanged_terms(units_list=hidden_units,
                                                    vmap=chain_pair_v)
        h_fe_pair = rbm.free_energy_affected_terms_from_activation(units_list=hidden_units,
                                                    vmap=chain_pair)
        fe_pair = dict((v, v_fe_pair[v] * beta[rv_i:rv_i+2] + h_fe_pair)
                       for v, tensor in v_fe_pair.keys())
        fe1, fe2 = fe_pair[rbm.v] # looses flexibility
        b1, b2 = beta[rv_i], beta[rv_i+1]
        r = T.exp((b1 - b2) * (fe1 - fe2))
        if rv_u < r:
            indices = T.arange(N_chains)
            indices[rv_i] = rv_i + 1
            indices[rv_i+1] = rv_i
            v1_gibbs_vmap = manipulate_vmap(v1_gibbs_vmap_preswap,
                                                 lambda x: x[indices, :])
            h1_gibbs_vmap = manipulate_vmap(h1_gibbs_vmap_preswap,
                                                 lambda x: x[indices, :])
            v1_stats_vmap = manipulate_vmap(v1_stats_vmap_preswap,
                                            lambda x: x[indices[:m], :])
        
        
        h1_stats_vmap = manipulate_vmap(h1_gibbs_vmap, lambda x: x[:m, :] * rescale)
        
        # get the v1 values in a fixed order
        v1_activation_values = [v1_activation_vmap[u] for u in visible_units]
        v1_gibbs_values = [v1_gibbs_vmap[u] for u in visible_units]
        v1_stats_values = [v1_stats_vmap[u] for u in visible_units]
        
        # same for the h1 values
        h1_activation_values = [h1_activation_vmap[u] for u in hidden_units]
        h1_gibbs_values = [h1_gibbs_vmap[u] for u in hidden_units]
        h1_stats_values = [h1_stats_vmap[u] for u in hidden_units]
        
        return v1_activation_values + v1_stats_values + v1_gibbs_values + \
               h1_activation_values + h1_stats_values + h1_gibbs_values
    
    
    chain_start = [persistent_vmap[u] for u in hidden_units]
    
    
    # The 'outputs_info' keyword argument of scan configures how the function outputs are mapped to the inputs.
    # in this case, we want the h1_gibbs_vmap values to map onto the function arguments, so they become
    # h0_gibbs_vmap values in the next iteration. To this end, we construct outputs_info as follows:
    outputs_info = [None] * (len(exp_input)*3) + [None] * (len(exp_latent)*2) + list(chain_start)
    # 'None' indicates that this output is not used in the next iteration.
    
    exp_output_all_list, theano_updates = theano.scan(gibbs_hvh, outputs_info = outputs_info, n_steps = k)
    # we only need the final outcomes, not intermediary values
    exp_output_list = [out[-1] for out in exp_output_all_list]
            
    # reconstruct vmaps from the exp_output_list.
    n_input, n_latent = len(visible_units), len(hidden_units)
    vk_activation_vmap = dict(zip(visible_units, exp_output_list[0:1*n_input]))
    vk_stats_vmap = dict(zip(visible_units, exp_output_list[1*n_input:2*n_input]))
    vk_gibbs_vmap = dict(zip(visible_units, exp_output_list[2*n_input:3*n_input]))
    hk_activation_vmap = dict(zip(hidden_units, exp_output_list[3*n_input:3*n_input+1*n_latent]))
    hk_stats_vmap = dict(zip(hidden_units, exp_output_list[3*n_input+1*n_latent:3*n_input+2*n_latent]))
    hk_gibbs_vmap = dict(zip(hidden_units, exp_output_list[3*n_input+2*n_latent:3*n_input+3*n_latent]))
    
    # add the Theano updates for the persistent CD states:
    if persistent_vmap is not None:
        for u, v in persistent_vmap.items():
            theano_updates[v] = hk_gibbs_vmap[u] # this should be the gibbs vmap, and not the stats vmap!
    
    activation_data_vmap = v0_vmap.copy() # TODO: this doesn't really make sense to have in an activation vmap!
    activation_data_vmap.update(h0_activation_vmap)
    activation_model_vmap = vk_activation_vmap.copy()
    activation_model_vmap.update(hk_activation_vmap)
    
    stats = Stats(theano_updates) # create a new stats object
    
    # store the computed stats in a dictionary of vmaps.
    stats_data_vmap = v0_vmap.copy()
    stats_data_vmap.update(h0_stats_vmap)
    stats_model_vmap = vk_stats_vmap.copy()
    stats_model_vmap.update(hk_stats_vmap)
    stats.update({
      'data': stats_data_vmap,
      'model': stats_model_vmap,
    })
            
    stats['data_activation'] = activation_data_vmap
    stats['model_activation'] = activation_model_vmap
        
    return stats

