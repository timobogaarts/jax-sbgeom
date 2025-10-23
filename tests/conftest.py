# tests/conftest.py
import os
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--jax-platform",
        action="store",
        default=None,
        help="Set JAX platform (cpu/gpu/tpu) before import"
    )

    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )    

def pytest_configure(config):
    platform = config.getoption("--jax-platform")
    if platform:
        os.environ["JAX_PLATFORM_NAME"] = platform
        

def pytest_runtest_setup(item):
    # Check if the test has the "slow" marker
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("Skipping slow test, use --runslow to run")