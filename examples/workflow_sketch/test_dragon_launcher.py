import numpy as np

from smartredis import Client
from smartsim import Experiment
from smartsim.settings import DragonRunSettings



def main():
    # Set up database and start it
    PORT = 6780
    launcher = 'dragon'
    exp = Experiment('MSR', launcher=launcher)
    runArgs = {"np": 1, "ppn": 1, "cpu-bind": "numa"}
    kwargs = {
        'maxclients': 100000,
        'threads_per_queue': 4, # set to 4 for improved performance
        'inter_op_parallelism': 1,
        'intra_op_parallelism': 4,
        'cluster-node-timeout': 30000,
        }
    db = exp.create_database(
                             #port=PORT,
                             batch=False,
                             db_nodes=1,
                             #run_command=run_command,
                             interface=['hsn0'],
                             #hosts=dbNodes_list,
                             #run_args=runArgs,
                             single_cmd=True,
                             **kwargs
                            )


    exp.generate(db)
    print("\nStarting database ...")
    exp.start(db)
    print("Done\n")

    # Initialize SmartRedis client
    client = Client(address=db.get_address()[0], cluster=False)

    # Set the distributions of the input variables
    vel_dist_params = np.array([0.0, 1.0, 1.3])
    client.put_tensor('vel_dist',vel_dist_params)

    # get data
    print(f'vel_dist = {client.get_tensor("vel_dist")}')

    # Launch something
    SSDB = db.get_address()[0]
    nrs_settings = DragonRunSettings('python',
                                         exe_args="/eagle/datascience/balin/Nek/nekRS-ML_ConvReac/examples/workflow_sketch/nrs.py",
                                         run_args=None,
                                         env_vars={'SSDB' : SSDB})
    nrs_settings.set_nodes(1)
    nrs_settings.set_tasks_per_node(1)
    nrs_model = exp.create_model(f"nekrs", nrs_settings)
    exp.generate(nrs_model, overwrite=True)
    exp.start(nrs_model, summary=True, block=True)

if __name__=='__main__':
    main()

