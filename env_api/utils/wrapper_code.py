class WrappersCode:
            
    wrapper_cpp_template = """#include "Halide.h"
    #include "$func_name$_wrapper.h"
    #include "tiramisu/utils.h"
    #include <iostream>
    #include <time.h>
    #include <fstream>
    #include <chrono>

    using namespace std::chrono;
    using namespace std;

    int main(int, char **argv){
            
    $buffers_init$
        
        //halide_set_num_threads(48);
        
        int nb_execs = get_nb_exec();

        double duration;
        
        for (int i = 0; i < nb_execs; ++i) {
            auto begin = std::chrono::high_resolution_clock::now(); 
            $func_name$($func_params$);
            auto end = std::chrono::high_resolution_clock::now(); 

            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000;
            std::cout << duration << " "; 

        }
        std::cout << std::endl;
        return 0;
    }"""
    wrapper_h_template = """#include <tiramisu/utils.h>
    #include <sys/time.h>
    #include <cstdlib>
    #include <algorithm>
    #include <vector>

    #define NB_THREAD_INIT 48
    struct args {
        double *buf;
        unsigned long long int part_start;
        unsigned long long int part_end;
        double value;
    };

    void *init_part(void *params)
    {
    double *buffer = ((struct args*) params)->buf;
    unsigned long long int start = ((struct args*) params)->part_start;
    unsigned long long int end = ((struct args*) params)->part_end;
    double val = ((struct args*) params)->value;
    for (unsigned long long int k = start; k < end; k++){
        buffer[k]=val;
    }
    pthread_exit(NULL);
    }

    void parallel_init_buffer(double* buf, unsigned long long int size, double value){
        pthread_t threads[NB_THREAD_INIT]; 
        struct args params[NB_THREAD_INIT];
        for (int i = 0; i < NB_THREAD_INIT; i++) {
            unsigned long long int start = i*size/NB_THREAD_INIT;
            unsigned long long int end = std::min((i+1)*size/NB_THREAD_INIT, size);
            params[i] = (struct args){buf, start, end, value};
            pthread_create(&threads[i], NULL, init_part, (void*)&(params[i])); 
        }
        for (int i = 0; i < NB_THREAD_INIT; i++) 
            pthread_join(threads[i], NULL); 
        return;
    }
    #ifdef __cplusplus
    extern "C" {
    #endif
    int $func_name$($func_params$);
    #ifdef __cplusplus
    }  // extern "C"
    #endif


    int get_beam_size(){
        if (std::getenv("BEAM_SIZE")!=NULL)
            return std::stoi(std::getenv("BEAM_SIZE"));
        else{
            std::cerr<<"error: Environment Variable BEAM_SIZE not declared"<<std::endl;
            exit(1);
        }
    }

    int get_max_depth(){
        if (std::getenv("MAX_DEPTH")!=NULL)
            return std::stoi(std::getenv("MAX_DEPTH"));
        else{
            std::cerr<<"error: Environment Variable MAX_DEPTH not declared"<<std::endl;
            exit(1);
        }
    }

    void declare_memory_usage(){
        setenv("MEM_SIZE", std::to_string((double)(256*192+320*256+320*192)*8/1024/1024).c_str(), true); // This value was set by the Code Generator
    }

    int get_nb_exec(){
        if (std::getenv("NB_EXEC")!=NULL)
            return std::stoi(std::getenv("NB_EXEC"));
        else{
            return 30;
        }
    }
    """