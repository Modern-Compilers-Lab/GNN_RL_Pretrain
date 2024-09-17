from env_api.tiramisu_api import TiramisuEnvAPI
from config.config import Config
import random, traceback, time

if __name__ == "__main__":
    start = time.time()
    # Init global config to run the Tiramisu env
    Config.init()
    tiramisu_api = TiramisuEnvAPI(local_dataset=True)
    # Get a list of the program names in the database
    programs = tiramisu_api.get_programs()
    try:
        # Select a program randomly for example program = "function025885"
        # program: str = random.choice(programs)
        program = "function_mvt_MEDIUM"
        print("Selected function : ", program)

        # set_program(str) creates all the necessary objects to start doing operations on a program
        # it returns an encoded representation specific to the RL system
        # This representation has a shape and type of torch.Size([180])
        actions_mask = tiramisu_api.set_program(name=program)
        
        # There is some programs that are not supported so we need to check our representation first
        if True:
            # After setting a program and checking if it is fully supported by our RL system, you can apply any action on it in any order
            # And expect to get the speedup of the whole schedule, the representation and the result of legality check of the last operation
            
            # Skewing
            # (speedup, embedding_tensor,
            #  legality,actions_mask) = tiramisu_api.skew(loop_level1=0,loop_level2=1,env_id=2)
            # (speedup,
            #  legality,actions_mask) = tiramisu_api.skew(loop_level1=0,loop_level2=1,env_id=2)
            
            # Interchange
            # (
            #     speedup,
            #     legality,
            #     actions_mask,
            # ) = tiramisu_api.interchange(loop_level1=1,loop_level2=2, env_id=7)
            
            # Parallelization
            (
                speedup,
                legality,
                actions_mask,
            ) = tiramisu_api.parallelize(loop_level=0, env_id=1)
            print("Action: Parallelization(0,1)")
            print(f"Speedup: {speedup:.3f}")
            print(f"Legality: {legality}")
            
            # Next Branch
            # tiramisu_api.scheduler_service.next_branch()
            
            # Tiling
            # (speedup, legality, actions_mask) = tiramisu_api.tile2D(
            #     loop_level1=1, loop_level2=2, size_x=6, size_y=6, env_id=4
            # )
            
            # Reversal
            # (
            #     speedup,
            #     legality,
            #     actions_mask,
            # ) = tiramisu_api.reverse(loop_level=2, env_id=1)
            
            # Unrolling
            # (speedup, legality, actions_mask,
            # ) = tiramisu_api.unroll(unrolling_factor=32, env_id=7, worker_id="1")

            # Next Branch
            # tiramisu_api.scheduler_service.next_branch()
            
            # Tiling
            (speedup,
             legality,actions_mask) = tiramisu_api.tile2D(loop_level1=0,loop_level2=1,size_x=32,size_y=64,env_id=4)
            print("Action: Tiling2D(0,1)")
            print(f"Speedup: {speedup:.3f}")
            print(f"Legality: {legality}")

            # # (speedup, embedding_tensor,
            # # legality,actions_mask) = tiramisu_api.tile3D(loop_level1=0 , loop_level2=1,loop_level3=2,
            # #     size_x=128,size_y=128,size_z=128,env_id=17)
            
        print(f"Time: {time.time() - start:.3f}")

    except Exception as e:
        print("Traceback of the error : " + 60 * "-")
        print(traceback.print_exc())
        print(80 * "-")
