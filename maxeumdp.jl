using Gen
import Statistics: mean

function softmax(x)
  exp.(x) / sum(exp.(x))
end

mu = [1.0, 0.0, 0.0, 0.0, 0.0]

# action state state
T = [
  [
    [0.0, 0.2, 0.6, 0.2, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
  ],
  [
    [0.0, 0.05, 0.9, 0.05, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
  ]]

# action state
R = [
  [0.0, -10.0, 6.0, 8.0, 0.0],
  [0.0, -10.0, 6.0, 8.0, 0.0],
]

num_act = length(T)
num_sta = length(T[1])

function getTransition(state, action)
  probs = T[action][state]
  return probs
end

function getUtility(state, action)
  return R[action][state]
end

function terminate(state)::Bool
  if state == 5
    return true
  else
    return false
  end
end

struct Und <: Distribution{Bool}
  lambda::Float64
end

const und(lambda) = Und(lambda)()

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

# policy
@gen function act(state, alpha::Float64)
  action ~ uniform_discrete(1, num_act)
  eu = inferER(state, action, alpha)
  # if state == 1
  #   if action == 1
  #     eu = 3.2
  #   else
  #     eu = 5.3
  #   end
  # else
  #   eu = 0
  # end

  @trace(Und(alpha * eu)(), :factor)
  return action
end

@gen function generateReturn(state, action, alpha::Float64)
  utility = getUtility(state, action)

  if terminate(state)
    return utility
  else
    next_state ~ categorical(getTransition(state, action))
    next_action ~ act(next_state, alpha)
    future_utility ~ generateReturn(next_state, next_action, alpha)
    total_utility = utility + future_utility
    return total_utility
  end
end

function inferER(state, action, alpha::Float64, num_samples::Int=10)
  observations = choicemap()
  traces, log_norm_weights, _ = importance_sampling(generateReturn, (state, action, alpha), observations, num_samples)
  expectation = sum(get_retval(tr) * exp(w) for (tr, w) in zip(traces, log_norm_weights))
  return expectation
end

@gen function simulate(alpha::Float64)
  state = {:state => 0} ~ categorical(mu)
  h_states = [state]
  h_actions = []
  h_rewards = []
  i = 1
  while !terminate(state)
    action = {:action => i} ~ act(state, alpha)
    # if state == 1
    #   p = softmax([alpha * 3.2, alpha * 5.3])
    #   action = {:action => i} ~ categorical(p)
    # else
    #   action = {:action => i} ~ categorical([0.5, 0.5])
    # end
    # action = @trace(act(state, alpha), :action => i)
    reward = getUtility(state, action)
    state = {:state => i} ~ categorical(getTransition(state, action))
    # state = categorical(getTransition(state, action))
    push!(h_actions, action)
    push!(h_rewards, reward)
    push!(h_states, state)
    i += 1
  end
  return (h_states, h_actions, h_rewards)
end

function inferAgentMH(num_iters::Int)
  args = (10.0,)
  trace, _ = generate(simulate, args)
  actions1 = Int[]
  actions2 = Int[]
  for i = 1:num_iters
    # selections = get_subselections(select(:action))
    selections = select(:action => 1 => :action, :action => 2 => :action, :state => 1, :state => 2)
    # selections = select(:action => 1 => :action)
    # selections = select()
    println(i)
    trace, _ = metropolis_hastings(trace, selections)
    println(get_retval(trace))
    push!(actions1, trace[:action=>1])
    push!(actions2, trace[:action=>2])
  end
  return (actions1, actions2)
end

function inferAgentIS(num_samples::Int=100)
  args = (100.0,)
  observations = choicemap()
  traces, log_norm_weights, _ = importance_sampling(simulate, args, observations, num_samples)
  s = logsumexp([get_score(tr) for tr in traces])
  total = sum(sum(get_retval(tr)[3]) * exp(get_score(tr) - s) for (tr, w) in zip(traces, log_norm_weights))
  return total
end

expected_return = inferAgentMH(100)
println("expected_return: ", expected_return)
