using Gen
import Statistics: mean

actions = ["italian", "french"]
action_probs = [0.5, 0.5]

# state x action x state
T = [
  [
  [
    0.2, 0.6, 0.2
  ],
  [
    0.05, 0.9, 0.05
  ]
]
]

# state
R = [
  -10, 6, 8
]

function getTransition(state, action)
  probs = T[state][action]
  # println(probs)
  return probs
end

function getUtility(state)
  return R[state]
end

struct Und <: Distribution{Bool}
  lambda::Float64
end

function Gen.random(::Und)
  return true
end

function Gen.logpdf(d::Und, x::Bool)
  if x == true
    return d.lambda
  else
    return -Inf
  end
end

# function Gen.has_output_grad(d::Und)
#   return false
# end
#
# function Gen.logpdf_grad(d::Und, x::Bool)
#   return nothing
# end
#
# function Gen.has_argument_grads(d::Und)
#   return false
# end

@gen function expectedUtility(state, action)
  # println("$state, $action")
  next_state ~ categorical(getTransition(state, action))
  utility = getUtility(next_state)
  return utility
end

function inferUtility(state, action, num_samples=100)
  observations = choicemap()
  traces, log_norm_weights, _ = importance_sampling(expectedUtility, (state, action), observations, num_samples)
  expectation = sum(get_retval(tr) * exp(w) for (tr, w) in zip(traces, log_norm_weights))
  return expectation
end

@gen function softMaxAgent(state, alpha=1.0)
  # TODO why does the inference making state a float?
  state = Int(state)
  action ~ categorical(action_probs)

  utility = inferUtility(state, action)

  und = Und(alpha * utility)
  @trace(und(), :factor)

  return action
end

function inferAgent(num_iters::Int)
  initial_state = 1
  trace, _ = generate(softMaxAgent, (initial_state, 1.0))
  actions = Int[]
  for _ = 1:num_iters
    trace, _ = metropolis_hastings(trace, select(:action))
    push!(actions, trace[:action])
  end
  return actions
end

# function inferAgentIS(num_samples::Int)
#   initial_state = 1
#
#   observations = choicemap()
#   traces, log_norm_weights, _ = importance_sampling(simulate, args, observations, num_samples)
#   for trace in traces
#     # println("states: ", get_retval(trace)[1])
#     # println("actions: ", get_retval(trace)[2])
#     # println("rewards: ", get_retval(trace)[3])
#   end
#   # expectation = sum(sum(get_retval(tr)[3]) * exp(w) for (tr, w) in zip(traces, log_norm_weights))


using Plots
actions = inferAgent(1000)
histogram(actions, label="α=1.0")
# println("average action: ", actions)
