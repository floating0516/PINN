import pytest
import torch


def test_constant_velocity_provider_returns_p_s_and_relative_delays() -> None:
    from src.physics.travel_time import ConstantVelocityTravelTime

    provider = ConstantVelocityTravelTime(
        alpha_m_per_s=8.0,
        beta_m_per_s=4.0,
    )

    delays = provider.delays(torch.tensor([8.0, 16.0]))

    assert torch.equal(delays.p_sec, torch.tensor([1.0, 2.0]))
    assert torch.equal(delays.s_sec, torch.tensor([2.0, 4.0]))
    assert torch.equal(delays.s_after_p_sec, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize(
    ("alpha", "beta", "message"),
    [
        (0.0, 4.0, "alpha_m_per_s"),
        (8.0, -1.0, "beta_m_per_s"),
        (float("nan"), 4.0, "alpha_m_per_s"),
    ],
)
def test_constant_velocity_provider_rejects_invalid_velocities(
    alpha: float,
    beta: float,
    message: str,
) -> None:
    from src.physics.travel_time import ConstantVelocityTravelTime

    with pytest.raises(ValueError, match=message):
        ConstantVelocityTravelTime(
            alpha_m_per_s=alpha,
            beta_m_per_s=beta,
        )


def test_provider_factory_uses_the_active_config_velocities() -> None:
    from src.physics.travel_time import travel_time_from_config

    provider = travel_time_from_config(
        {
            "physics": {
                "travel_time_model": "constant_velocity",
                "alpha": 7900.0,
                "beta": 4533.0,
            }
        }
    )

    assert provider.alpha_m_per_s == 7900.0
    assert provider.beta_m_per_s == 4533.0


def test_provider_factory_rejects_unknown_model() -> None:
    from src.physics.travel_time import travel_time_from_config

    with pytest.raises(ValueError, match="travel_time_model"):
        travel_time_from_config(
            {
                "physics": {
                    "travel_time_model": "crust1",
                    "alpha": 7900.0,
                    "beta": 4533.0,
                }
            }
        )
