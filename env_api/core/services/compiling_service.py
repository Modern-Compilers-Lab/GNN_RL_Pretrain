import subprocess
import re, copy, logging
from typing import List
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.scheduler.models.action import Action, Parallelization, Tiling, Unrolling
from env_api.scheduler.models.branch import Branch
from env_api.scheduler.models.schedule import Schedule
from config.config import Config
from pathlib import Path


class CompilingService:
    @classmethod
    def compile_legality(
        cls,
        schedule_object: Schedule,
        optims_list: List[Action],
    ):
        tiramisu_program = schedule_object.prog
        output_path = f"{Config.config.tiramisu.workspace}{tiramisu_program.name}legal{optims_list[-1].worker_id}"

        cpp_code = cls.get_legality_code(schedule_object=schedule_object, optims_list=optims_list)

        return cls.run_cpp_code(cpp_code=cpp_code, output_path=output_path)

    @classmethod
    def get_legality_code(
        cls,
        schedule_object: Schedule,
        optims_list: List[Action],
    ):
        
        cpp_code = schedule_object.prog.original_str
        comps_dict = {}
        for comp in schedule_object.prog.annotations["computations"]:
            comps_dict[comp] = copy.deepcopy(schedule_object.prog.annotations["computations"][comp]["iterators"])
        # Add code to the original file to get legality result
        legality_check_lines = """
        prepare_schedules_for_legality_checks(true);
        perform_full_dependency_analysis();
        bool is_legal=true;"""

        updated_fusion = ""
        unrolling_legality = ""
        tiling_in_actions = False

        for optim in optims_list :
            if isinstance(optim, Unrolling): 
                unrolling_legality += optim.legality_code_str
            else : 
                legality_check_lines += optim.legality_code_str
            
            if isinstance(optim, Tiling):
                tiling_in_actions = True
                loop_levels_size = len(optim.params) // 2
                #  Add the tiling new loops to comps_dict
                for impacted_comp in optim.comps:
                    for loop_index in optim.params[:loop_levels_size]:
                        comps_dict[impacted_comp].insert(
                            loop_levels_size + loop_index, f"t{loop_index}"
                        )


        if (tiling_in_actions):
            updated_fusion, cpp_code = cls.fuse_tiling_loops(
                code=cpp_code, comps_dict=comps_dict
            )
            legality_check_lines += "\n\tclear_implicit_function_sched_graph();"
        


        legality_check_lines += f"""
            {updated_fusion}
            {unrolling_legality}
            prepare_schedules_for_legality_checks(true);
            is_legal &= check_legality_of_function();   
            std::cout << is_legal;
            """

        # Paste the lines responsable of checking legality of schedule in the cpp file
        cpp_code = cpp_code.replace(
            schedule_object.prog.code_gen_line, legality_check_lines
        )
        return cpp_code


    @classmethod
    def run_cpp_code(cls, cpp_code: str, output_path: str):
        shell_script = [
            f"CXX={Config.config.env_vars.CXX}",
            f"CONDA_ENV={Config.config.env_vars.CONDA_ENV}",
            f"TIRAMISU_ROOT={Config.config.env_vars.TIRAMISU_ROOT}",
            f"LD_LIBRARY_PATH={Config.config.env_vars.LD_LIBRARY_PATH}",
            "export CXX",
            "export CONDA_ENV",
            "export TIRAMISU_ROOT",
            "export LD_LIBRARY_PATH",
            # Compile intermidiate tiramisu file
            "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/build/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/build/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -fopenmp -std=c++17 -O0 -o {}.o -c -x c++ -".format(
                output_path
            ),
            # Link generated file with executer
            "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -fopenmp -std=c++17 -O0 {}.o -o {}.out -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/build/src  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/build/src:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl".format(
                output_path, output_path
            ),
            # Run the program
            "{}.out".format(output_path),
            # Clean generated files
            "rm {}.out {}.o".format(output_path, output_path),
        ]
        try:
            compiler = subprocess.run(
                ["\n".join(shell_script)],
                input=cpp_code,
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )
            return compiler.stdout if compiler.stdout != "" else "0"
        except subprocess.CalledProcessError as e:
            print("Process terminated with error code", e.returncode)
            print("Error output:", e.stderr)
            return "0"
        except Exception as e:
            logging.log(e)
            return "0"

    @classmethod
    def call_skewing_solver(cls, schedule_object, optim_list, action: Action):
        params = action.params
        legality_cpp_code = cls.get_legality_code(schedule_object, optim_list)
        to_replace = re.findall(r"std::cout << is_legal;", legality_cpp_code)[0]
        legality_cpp_code = legality_cpp_code.replace(
            "is_legal &= check_legality_of_function();", ""
        )
        legality_cpp_code = legality_cpp_code.replace("bool is_legal=true;", "")
        legality_cpp_code = re.sub(
            r"is_legal &= loop_parallelization_is_legal.*\n", "", legality_cpp_code
        )
        legality_cpp_code = re.sub(
            r"is_legal &= loop_unrolling_is_legal.*\n", "", legality_cpp_code
        )

        solver_lines = (
            "function * fct = tiramisu::global::get_implicit_function();\n"
            + "\n\tauto auto_skewing_result = fct->skewing_local_solver({"
            + ", ".join([f"&{comp}" for comp in action.comps])
            + "}"
            + ",{},{},1);\n".format(*params)
        )

        solver_lines += """    
        std::vector<std::pair<int,int>> outer1, outer2,outer3;
        tie( outer1,  outer2,  outer3 )= auto_skewing_result;
        if (outer1.size()>0){
            std::cout << outer1.front().first;
            std::cout << ",";
            std::cout << outer1.front().second;
            std::cout << ",";
        }else {
            std::cout << "None,None,";
        }
        if(outer2.size()>0){
            std::cout << outer2.front().first;
            std::cout << ",";
            std::cout << outer2.front().second;
            std::cout << ",";
        }else {
            std::cout << "None,None,";
        }
        if(outer3.size()>0){
            std::cout << outer3.front().first;
            std::cout << ",";
            std::cout << outer3.front().second;
            std::cout << ",";
        }else {
            std::cout << "None,None,";
        }
        
            """

        solver_code = legality_cpp_code.replace(to_replace, solver_lines)
        output_path = (
            Config.config.tiramisu.workspace
            + schedule_object.prog.name
            + "skew_solver"
            + action.worker_id
        )
        result_str = cls.run_cpp_code(cpp_code=solver_code, output_path=output_path)
        if not result_str:
            return None
            # Refer to function run_cpp_code to see from where the "0" comes from
        elif result_str == "0":
            return None
        result_str = result_str.split(",")
        # Skewing Solver returns 3 solutions in form of tuples, the first tuple is for outer parallelism ,
        # second is for inner parallelism , and last one is for locality, we are going to use the first preferably
        # if availble , else , we are going to use the scond one if available, this policy of choosing factors may change
        # in later versions!
        # The compiler in our case returns a tuple of type : (fac0,fac1,fac2,fac3,fac4,fac5) each 2 factors represent the
        # solutions mentioned above
        if result_str[0] != "None":
            # Means we have a solution for outer parallelism
            fac1 = int(result_str[0])
            fac2 = int(result_str[1])
            return fac1, fac2
        if result_str[2] != "None":
            # Means we have a solution for inner parallelism
            fac1 = int(result_str[2])
            fac2 = int(result_str[3])
            return fac1, fac2
        else:
            return None

    @classmethod
    def fuse_tiling_loops(cls, code: str, comps_dict: dict):
        fusion_code = ""
        # This pattern will detect lines that looks like this :
        # ['comp00.then(comp01, i2)',
        # '.then(comp02, i1)',
        # '.then(comp03, i3)',
        # '.then(comp04, i1);']
        regex_first_comp = r"(\w+)\.then\("
        matching = re.search(regex_first_comp, code)

        if matching is None:
            return fusion_code, code

        # comps will contain all the computations that are fused together
        comps = [matching.group(1)]

        # regex rest of the thens
        regex_rest = r"\.then\(([\w]+),"
        # results will contain all the lines that match the regex
        for result in re.findall(regex_rest, code):
            comps.append(result)

        # levels indicates which loop level the 2 comps will be seperated in
        levels = []
        # updated_lines will contain new lines of code with the new seperated levels
        updated_lines = []
        # Defining intersection between comps' iterators
        for i in range(len(comps) - 1):
            level = 0
            while True:
                if comps_dict[comps[i]][level] == comps_dict[comps[i + 1]][level]:
                    if (
                        level + 1 == comps_dict[comps[i]].__len__()
                        or level + 1 == comps_dict[comps[i + 1]].__len__()
                    ):
                        levels.append(level)
                        break
                    level += 1
                else:
                    levels.append(level - 1)
                    break
            if levels[-1] == -1:
                updated_lines.append(f".then({comps[i+1]}, computation::root)")
            else:
                updated_lines.append(f".then({comps[i+1]},{levels[-1]})")

        updated_lines[0] = comps[0] + updated_lines[0]
        updated_lines[-1] = updated_lines[-1] + ";"

        for line in range(len(comps) - 1):
            # code = code.replace(results[line],"")
            fusion_code += updated_lines[line]

        return fusion_code, code

    @classmethod
    def get_schedule_code(
        cls,
        tiramisu_program: TiramisuProgram,
        optims_list: List[Action],
    ):
        cpp_code = tiramisu_program.original_str
        comps_dict = {}
        for comp in tiramisu_program.annotations["computations"]:
            comps_dict[comp] = copy.deepcopy(tiramisu_program.annotations["computations"][comp]["iterators"])

        updated_fusion = ""
        unrolling_updated = ""
        schedule_code = ""
        tiling_in_actions = False

        for optim in optims_list :
            if isinstance(optim, Unrolling): 
                unrolling_updated += optim.execution_code_str
            else : 
                schedule_code += optim.execution_code_str
            
            if isinstance(optim, Tiling):
                tiling_in_actions = True
                loop_levels_size = len(optim.params) // 2
                #  Add the tiling new loops to comps_dict
                for impacted_comp in optim.comps:
                    for loop_index in optim.params[:loop_levels_size]:
                        comps_dict[impacted_comp].insert(
                            loop_levels_size + loop_index, f"t{loop_index}"
                        )

        if (tiling_in_actions):
            updated_fusion, cpp_code = cls.fuse_tiling_loops(
                code=cpp_code, comps_dict=comps_dict
            )
            schedule_code += "\n\tclear_implicit_function_sched_graph();"

        schedule_code += f"""
            {updated_fusion}
            {unrolling_updated}
            """

        # Add code gen line to the schedule code
        schedule_code += "\n\t" + tiramisu_program.code_gen_line + "\n"
        # Paste the lines responsable of checking legality of schedule in the cpp file
        cpp_code = tiramisu_program.original_str.replace(
            tiramisu_program.code_gen_line, schedule_code
        )
        cpp_code = cpp_code.replace(
            f"// {tiramisu_program.wrapper_str}", tiramisu_program.wrapper_str
        )
        return cpp_code

    @classmethod
    def write_cpp_code(cls, cpp_code: str, output_path: str):
        with open(output_path + ".cpp", "w") as f:
            f.write(cpp_code)

    @classmethod
    def execute_code(
        cls,
        tiramisu_program: TiramisuProgram,
        optims_list: List[Action],
        timeout: int = None
    ):
        if not optims_list: 
            worker = "init"
        else : 
            worker = str(optims_list[-1].worker_id)

        path = Path().joinpath(Config.config.tiramisu.workspace, worker)

        if not path.exists():
            path.mkdir(parents=True)

        execution_time = None

        cpp_code = cls.get_schedule_code(
            tiramisu_program=tiramisu_program,
            optims_list=optims_list,
        )

        output_path = f"{str(path)}/{tiramisu_program.name}"

        cpp_file_path = output_path + f"_schedule.cpp" 
        with open(cpp_file_path, "w") as file:
            file.write(cpp_code)

        wrapper_cpp, wrapper_h = tiramisu_program.build_wrappers()

        wrapper_cpp_path = output_path + f"_wrapper.cpp"
        wrapper_h_path = output_path + f"_wrapper.h"

        with open(wrapper_cpp_path, "w") as file:
            file.write(wrapper_cpp)

        with open(wrapper_h_path, "w") as file:
            file.write(wrapper_h)

        object_name = f"{tiramisu_program.name}.o"
        out_name = f"{tiramisu_program.name}.out"

        wrapper_exec = f"{tiramisu_program.name}_wrapper"

        shell_script = [
            f"CXX={Config.config.env_vars.CXX}",
            f"CONDA_ENV={Config.config.env_vars.CONDA_ENV}",
            f"TIRAMISU_ROOT={Config.config.env_vars.TIRAMISU_ROOT}",
            f"LD_LIBRARY_PATH={Config.config.env_vars.LD_LIBRARY_PATH}",
            "export CXX",
            "export CONDA_ENV",
            "export TIRAMISU_ROOT",
            "export LD_LIBRARY_PATH",
            f"cd {str(path)}",
            # Compile intermidiate tiramisu file
            f"$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/build/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/build/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -fopenmp -std=c++17 -O0 -o {object_name} -c {cpp_file_path}",
            # Link generated file with executer
            f"$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -fopenmp -std=c++17 -O0 {object_name} -o {out_name}  -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/build/src  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/build/src:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl",
            # Run the generator
            f"./{out_name}",
            # compile the wrapper
            f"$CXX -shared -o {object_name}.so {object_name}",
            f"$CXX -std=c++17 -fno-rtti -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/Halide/build/include -I$TIRAMISU_ROOT/3rdParty/isl/include/ -I$TIRAMISU_ROOT/benchmarks -L$TIRAMISU_ROOT/build -L$TIRAMISU_ROOT/3rdParty/Halide/build/src -L$TIRAMISU_ROOT/3rdParty/isl/build/lib -o {wrapper_exec} -ltiramisu -lHalide -ldl -lpthread -fopenmp -lm -Wl,-rpath,$TIRAMISU_ROOT/build {wrapper_cpp_path} ./{object_name}.so -ltiramisu -lHalide -ldl -lpthread -fopenmp -lm -lisl",
        ]
        try:
            # print("#"*10, "COMPILING INTERMEDIATE TIRAMISU, GENERATOR, WRAPPER", "#"*10)
            compiler = subprocess.run(
                [" \n ".join(shell_script)],
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Process terminated with error code: {e.returncode}")
            logging.error(f"Error output: {e.stderr}")
            logging.error(f"Output: {e.stdout}")

        except subprocess.TimeoutExpired as e :
            logging.error("Timeout Error")
            raise subprocess.TimeoutExpired(compiler.args, timeout)

        except Exception as e:
            logging.error(e)
        
        run_script = [
            f"CXX={Config.config.env_vars.CXX}",
            f"CONDA_ENV={Config.config.env_vars.CONDA_ENV}",
            f"TIRAMISU_ROOT={Config.config.env_vars.TIRAMISU_ROOT}",
            f"LD_LIBRARY_PATH={Config.config.env_vars.LD_LIBRARY_PATH}",
            "export CXX",
            "export CONDA_ENV",
            "export TIRAMISU_ROOT",
            "export LD_LIBRARY_PATH",
            # cd to the workspace
            f"cd {str(path)}",
            # set the env variables
            f"export DYNAMIC_RUNS={Config.config.experiment.DYNAMIC_RUNS}",
            f"export MAX_RUNS={Config.config.experiment.MAX_RUNS}",
            f"export NB_EXEC={Config.config.experiment.NB_EXEC}",
            # run the wrapper
            f"./{wrapper_exec}",
        ]
        try:
            # print("#"*10, "RUNNING PROGRAM", "#"*10)
            compiler = subprocess.run(
                [" ; ".join(run_script)],
                capture_output=True,
                text=True,
                shell=True,
                check=True,
                timeout=timeout
            )
            numbers = compiler.stdout.split("\n")[-2].split(" ")[:-1]
            for i in range(len(numbers)):
                numbers[i] = float(numbers[i])
            if numbers:
                execution_time = min(numbers)

        except subprocess.CalledProcessError as e:
            logging.error(f"Process terminated with error code: {e.returncode}")
            logging.error(f"Error output: {e.stderr}")
            logging.error(f"Output: {e.stdout}")
            
        except subprocess.TimeoutExpired as e :
            logging.error("Timeout Error")
            raise subprocess.TimeoutExpired(compiler.args, timeout)
        
        except Exception as e:
            logging.error(e)

        # Deleting all files of the program
        try:
            subprocess.run(
                    [f"rm {output_path}*"],
                    capture_output=True,
                    text=True,
                    shell=True,
                    check=True,
                )
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Process terminated with error code: {e.returncode}")
            logging.error(f"Error output: {e.stderr}")
            logging.error(f"Output: {e.stdout}")
            
        except subprocess.TimeoutExpired as e :
            logging.error("Timeout Error")
            raise subprocess.TimeoutExpired(compiler.args, timeout)
        
        except Exception as e:
            logging.error(e)
            
        return execution_time

    @classmethod
    def compile_annotations(cls, tiramisu_program):
        # TODO : add getting tree structure object from executing the file instead of building it
        output_path = Config.config.tiramisu.workspace + tiramisu_program.name + "annot"
        # Add code to the original file to get json annotations

        get_json_lines = """
            auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function(), {});
            std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
            std::cout << program_json;
            """

        # Paste the lines responsable of generating the program json tree in the cpp file
        cpp_code = tiramisu_program.original_str.replace(
            tiramisu_program.code_gen_line, get_json_lines
        )
        return cls.run_cpp_code(cpp_code=cpp_code, output_path=output_path)