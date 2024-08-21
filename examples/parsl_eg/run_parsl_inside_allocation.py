import parsl
from parsl.config import Config
from parsl.providers import LocalProvider
#from parsl.launchers import SrunLauncher
from parsl.launchers import MpiExecLauncher
from parsl.channels import LocalChannel
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# Uncomment this to see logging info
#import logging
#logging.basicConfig(level=logging.DEBUG)

config = Config(
    executors=[
        HighThroughputExecutor(
            label='PM_HTEX_headless',
            # one worker per manager / node
            max_workers=1,
            cores_per_worker=32.0,
            provider=LocalProvider(
                channel=LocalChannel(script_dir='.'),
                # make sure the nodes_per_block mat
                nodes_per_block=2,
                #launcher=SrunLauncher(overrides='-c 32'),
                #launcher=MpiExecLauncher(overrides='-c 32'),
                cmd_timeout=120,
                init_blocks=1,
                max_blocks=1
            ),
        )
    ],
    strategy=None,
)

parsl.load(config)

from parsl import python_app
# Here we sleep for 2 seconds and return platform information
@python_app
def platform():
    import platform
    import time
    time.sleep(2)
    return platform.uname()

calls = [platform() for i in range(4)]
print(calls)

for c in calls:
    print("Got result: ", c.result())
