import numpy as np

from agent_experience_buffer import AgentReplayMemory


def test_agent_replay_memory():
    buffer = AgentReplayMemory(100, 2, 3, None)
    assert buffer.states.shape[0]*buffer.states.shape[1] >= 100
    c = 50
    assert buffer.states.shape == (c, 2, 3)
    assert buffer.states.shape == buffer.next_states.shape
    assert buffer.rewards.shape == (c, 2, 1)
    assert buffer.actions.shape == (c, 2, 1)
    assert buffer.dones.shape == (c, 2, 1)

    for i in range(c):
        assert len(buffer) == i
        assert not buffer.at_capacity()
        buffer.push_agent_actions(
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([False, False])
        )

    assert buffer.at_capacity()

    buffer.push_agent_actions(
        np.array([[11, 12, 13], [14, 15, 16]]),
        np.array([1, 2]),
        np.array([3, 4]),
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([False, False])
    )

    assert (buffer.states[0] == np.array([[11, 12, 13], [14, 15, 16]])).all()
    assert len(buffer) == c
