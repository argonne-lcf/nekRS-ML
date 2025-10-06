import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
import os 

plt.rcParams.update({'font.size': 22})

import torch

def get_grad_data(SIZE, keys, halo_mode_list):
    DATA_FULL = []
    for RANK in range(SIZE):
        data_rank = {}
        for halo_mode in halo_mode_list:
            model_name = 'RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar' %(RANK,SIZE,halo_mode)
            a = torch.load(data_path + '/' + model_name, map_location=torch.device('cpu'))
            
            # loss 
            loss = a['loss']
            
            # grad
            grad_full = torch.tensor([])
            for key in keys:
                grad = a[key].flatten()
                grad_full = torch.cat((grad_full, grad))

            data_rank[halo_mode] = {}
            data_rank[halo_mode]['loss'] = loss 
            data_rank[halo_mode]['grad'] = grad_full
            
        DATA_FULL.append(data_rank)
    return DATA_FULL 

if __name__ == "__main__":

    if 1 == 0:
        """
        Load a SINGLE model and plot its loss 
        """
        a = torch.load('saved_models/model.tar')
        loss_train = a['loss_hist_train']

        epochs = np.arange(1, len(loss_train)+1)


        fig, ax = plt.subplots()
        ax.plot(epochs, loss_train)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title('Training Demo -- Single Snapshot (Periodic Hill)')
        plt.show(block=False)

    if 1 == 0:
        """
        Model weight comparisons -- are the models the same? 
        """
        data_path = './outputs/postproc/models/tgv_poly_1'
        SIZE_LIST = [1,2,4,8,16]
        halo_mode = 'all_to_all'
        DATA_FULL = []
        for SIZE in SIZE_LIST:
            for RANK in range(SIZE):
                model_name = 'RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar' %(RANK,SIZE,halo_mode)
                a = torch.load(data_path + '/' + model_name, map_location=torch.device('cpu'))['state_dict']
                params_full = torch.tensor([])
                for key in a.keys():
                    param = a[key].flatten()
                    params_full = torch.cat((params_full, param))

                DATA_FULL.append(params_full.sum())

        fig, ax = plt.subplots()
        ax.plot(DATA_FULL, marker='o')
        plt.show(block=False)


    if 1 == 0:
        """
        Looking at consistency in training -- loss versus iter. 
        """
        POLY = 1
        #SIZE_LIST = [1,2,4,8]
        SIZE_LIST = [1,4]
        COLOR_LIST = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
        HALO_LIST = ['none', 'all_to_all', 'send_recv']
        #HALO_LIST = ['none']

        losses = [] 
        fig, ax = plt.subplots(figsize=(12,6))
        for i in range(len(SIZE_LIST)):
            for j in range(len(HALO_LIST)):
                size = SIZE_LIST[i]
                halo = HALO_LIST[j]
                if (size == 1) and (halo != 'none'):
                    continue 

                # # old gnn 
                # mp = 4 
                # model_path_1 = f"./saved_models/old_gnn/real_grad/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_{mp}_halo_{halo}.tar"
                # model_path_2 = f"./saved_models/old_gnn/hardcode_grad/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_{mp}_halo_{halo}.tar"

                # new gnn 
                mp = 4
                #model_path_1 = f"./saved_models/new_gnn/real_grad/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_3_7_32_3_2_{mp}_{halo}.tar"
                model_path_1 = f"./saved_models/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_3_4_32_3_2_{mp}_{halo}.tar"
                #model_path_2 = f"./saved_models/new_gnn/hardcode_grad/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_3_7_32_3_2_{mp}_{halo}.tar"

                a_1 = torch.load(model_path_1)
                loss_1 = a_1['loss_hist_train']

                #a_2 = torch.load(model_path_2)
                #loss_2 = a_2['loss_hist_train']

                losses.append(loss_1)

                if halo == 'none':
                    marker='o'
                    ls='-'
                elif halo == 'all_to_all':
                    marker='s'
                    ls='--'
                else:
                    marker='^'
                    ls='--'

                color = COLOR_LIST[i]
                ax.plot(np.arange(len(loss_1))+1, loss_1, 
                        marker=marker, color=color, ls=ls, mew=1., lw=1., ms=14, fillstyle='none',
                        label=f"{halo} -- {size} ranks")
                #ax.plot(loss_2, marker='s', color=color, ls=ls, lw=1, ms=15, fillstyle='none')

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend(fancybox=False, framealpha=1)
        #ax.set_xlim([1,10])

        plt.show(block=False)

    if 1 == 1:
        """
        Looking at profiler outputs (NEW -- custom timers) 
        """
        timer_dir = "./outputs/profiles/new_timers" 
        
        SIZE_LIST = [1,2,4]
        POLY = 1
        data_all_to_all = [] 
        data_send_recv = []
        data_none = []

        for SIZE in SIZE_LIST:
            data_all_to_all.append( torch.load(f"{timer_dir}/timers_avg_POLY_1_RANK_0_SIZE_{SIZE}_SEED_12_3_4_32_3_2_4_all_to_all.tar") )
            data_send_recv.append( torch.load(f"{timer_dir}/timers_avg_POLY_1_RANK_0_SIZE_{SIZE}_SEED_12_3_4_32_3_2_4_send_recv.tar") )
            data_none.append( torch.load(f"{timer_dir}/timers_avg_POLY_1_RANK_0_SIZE_{SIZE}_SEED_12_3_4_32_3_2_4_none.tar") )

        qoi = "forwardPass"
        qoi = "backwardPass"
        qoi = "loss"
        lb = 10
        fig, ax = plt.subplots()
        for i in range(len(SIZE_LIST)):
            ax.plot( SIZE_LIST[i], data_all_to_all[i][qoi][-lb:-1].mean(), marker='o', lw=0, ms=12, color='black' )  
            ax.plot( SIZE_LIST[i], data_all_to_all[i][qoi][-lb:-1].min(), marker='+', lw=0, ms=12, color='black' )  
            ax.plot( SIZE_LIST[i], data_all_to_all[i][qoi][-lb:-1].max(), marker='+', lw=0, ms=12, color='black' )  

            ax.plot( SIZE_LIST[i], data_send_recv[i][qoi][-lb:-1].mean(), marker='o', lw=0, ms=12, color='red' )
            ax.plot( SIZE_LIST[i], data_send_recv[i][qoi][-lb:-1].min(), marker='+', lw=0, ms=12, color='red' )
            ax.plot( SIZE_LIST[i], data_send_recv[i][qoi][-lb:-1].max(), marker='+', lw=0, ms=12, color='red' )

            ax.plot( SIZE_LIST[i], data_none[i][qoi][-lb:-1].mean(), marker='o', lw=0, ms=12, color='blue' )
            ax.plot( SIZE_LIST[i], data_none[i][qoi][-lb:-1].min(), marker='+', lw=0, ms=12, color='blue' )
            ax.plot( SIZE_LIST[i], data_none[i][qoi][-lb:-1].max(), marker='+', lw=0, ms=12, color='blue' )
        plt.show(block=False)

        asdf

    if 1 == 0: 
        """
        Looking at consistency QoIs -- data produced from train_step_verification in main.py  
        """
        # no cos(pos), no edge fix 
        # path_32 = "./outputs/postproc/real_gnn/periodic_after_fix/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        # path_64 = "./outputs/postproc/real_gnn/periodic_after_fix/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"

        # with pos=0, with edge fix 
        # path_32 = "./outputs/postproc/real_gnn_test/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        # path_64 = "./outputs/postproc/real_gnn_test/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"

        # with cos(pos), with edge fix 
        # path_32 = "./outputs/postproc/real_gnn_test_2/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        # path_64 = "./outputs/postproc/real_gnn_test_2/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"

        # with cos(pos), with edge fix, with binary read 
        # path_32 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        # path_64 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"

        # with cos(pos), with edge fix, with binary read -- polaris 
        # path_32 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_5/float32"
        # path_64 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_5/float32"

        # new gnn 
        path_32 = "./outputs/postproc/real_gnn_test_4/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_1/float32"
        path_64 = "./outputs/postproc/real_gnn_test_4/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_1/float32"

        SIZE_LIST = [1,2,4,8]
        #SIZE_LIST = [1,2,4,8,16,32] 
        #SIZE_LIST = [4,8,16,32] 
        #SIZE_LIST = [8,16,32] 
        HALO_MODE_LIST = ['none', 'all_to_all', 'send_recv']
        #HALO_MODE_LIST = ['all_to_all']
        #HALO_MODE_LIST = ['sendrecv']


        data_32 = {}
        data_64 = {}

        for halo_mode in HALO_MODE_LIST: 
            data_32[halo_mode] = []  
            data_64[halo_mode] = []  
            for SIZE in SIZE_LIST:
                data_temp_32 = np.zeros((SIZE,6))
                data_temp_64 = np.zeros((SIZE,6))
                for RANK in range(SIZE): 
                    
                    # # Toy gnn 
                    # str_temp = "TOY_RANK_%d_SIZE_%d_halo_%s.tar" %(RANK, SIZE, halo_mode) 

                    # Real gnn input channels 1 output channels 1  
                    # str_temp = "RANK_%d_SIZE_%d_input_channels_1_hidden_channels_1_output_channels_1_nMessagePassingLayers_5_halo_%s.tar" %(RANK, SIZE, halo_mode) 
                    
                    # Real gnn input channels 3 output channels 3
                    # str_temp = "RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar" %(RANK, SIZE, halo_mode) 

                    # New gnn format: 
                    mp = 4
                    seed = 12
                    str_temp = f"POLY_1_RANK_{RANK}_SIZE_{SIZE}_SEED_{seed}_3_4_32_3_2_{mp}_{halo_mode}.tar" 

                    a = torch.load(path_32 + "/" + str_temp, map_location=torch.device('cpu')) 
                    #data_temp_32[RANK, :3] = a['total_sum_x_scaled']
                    data_temp_32[RANK, :3] = a['total_sum_y_scaled']
                    #data_temp_32[RANK, :3] = a['total_sum_pos_scaled']
                    data_temp_32[RANK, 3] = a['effective_nodes']
                    data_temp_32[RANK, 4] = a['loss']
                    data_temp_32[RANK, 5] = a['effective_edges']

                    a = torch.load(path_64 + "/" + str_temp, map_location=torch.device('cpu')) 
                    #data_temp_64[RANK, :3] = a['total_sum_x_scaled']
                    data_temp_64[RANK, :3] = a['total_sum_y_scaled']
                    #data_temp_64[RANK, :3] = a['total_sum_pos_scaled']
                    data_temp_64[RANK, 3] = a['effective_nodes']
                    data_temp_64[RANK, 4] = a['loss']
                    data_temp_64[RANK, 5] = a['effective_edges']

                data_32[halo_mode].append(data_temp_32)
                data_64[halo_mode].append(data_temp_64)

        # Plot components 
        ms=250
        colors={'none': 'red', 'all_to_all': 'blue', 'send_recv': 'green'}
        ls={'none': '-', 'all_to_all': '-.', 'send_recv': '--'}
        fig, ax = plt.subplots(1,3,figsize=(16,5))
        for comp in range(3):
            for i in range(len(SIZE_LIST)): 
                for halo_mode in HALO_MODE_LIST: 
                    SIZE = SIZE_LIST[i]
                    ax[comp].scatter(np.ones(SIZE)*SIZE, data_32[halo_mode][i][:,comp], marker='^', 
                               color=colors[halo_mode], s=ms, facecolors='none',
                               linestyle=ls[halo_mode], linewidth=2,
                               label="CPU, FP32" if i == 0 else None)

                    ax[comp].scatter(np.ones(SIZE)*SIZE, data_64[halo_mode][i][:,comp], marker='s', 
                               color=colors[halo_mode], s=ms, facecolors='none', 
                               linestyle=ls[halo_mode], linewidth=2, 
                               label="CPU, FP64" if i == 0 else None)

                    ax[comp].set_title('Component %d' %(comp))
                    ax[comp].set_xlabel('Number of Ranks')

                    #ax[comp].set_ylim([0.0766, 0.0770])
                    ax[comp].set_xlim([0.9, 40])
                    ax[comp].set_xscale('log')
        #ax.set_xscale('log')
        #ax[0].legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 14})
        plt.show(block=False)

        # # Plot loss
        # ms=200
        # colors={'none': 'black', 'all_to_all': 'blue'}
        # fig, ax = plt.subplots(figsize=(6,5))
        # for i in range(len(SIZE_LIST)): 
        #     for halo_mode in HALO_MODE_LIST: 
        #         SIZE = SIZE_LIST[i]

        #         ax.scatter(np.ones(SIZE)*SIZE, data_32[halo_mode][i][:,4], marker='^', 
        #                    color=colors[halo_mode], s=ms, facecolors='none', 
        #                    label="CPU, FP32" if i == 0 else None)

        #         ax.scatter(np.ones(SIZE)*SIZE, data_64[halo_mode][i][:,4], marker='s', 
        #                    color=colors[halo_mode], s=ms, facecolors='none', 
        #                    label="CPU, FP64" if i == 0 else None)

        #         ax.set_title('Loss')
        #         ax.set_xlabel('Number of Ranks')

        # #ax.set_xscale('log')
        # #ax.legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 14})
        # ax.set_ylim([0.0763, 0.0768])
        # plt.show(block=False)


    if 1 == 0:
        """
        Looking at graph stats -- number of nodes, edges, etc. 
        """

        # POLY_LIST = [1, 3, 5, 7]  
        NELE_LIST = [8, 16, 20, 24, 32, 40, 48, 56, 64]
        #NELE_LIST = [16, 32, 64, 128, 256, 512]
        POLY_LIST = [5]  
        SIZE_LIST = [1, 2, 4, 8, 16, 32, 64] # 128]

        n_nodes_local_ele = []
        n_nodes_halo_ele = []
        n_edges_ele = []
        for e in range(len(NELE_LIST)):
            Nele = NELE_LIST[e]
            n_nodes_local = [] 
            n_nodes_halo = []
            n_edges = [] 
            for i in range(len(POLY_LIST)):
                POLY = POLY_LIST[i]
                n_nodes_local.append([])
                n_nodes_halo.append([])
                n_edges.append([])
                for j in range(len(SIZE_LIST)):
                    SIZE = SIZE_LIST[j]
                    n_nodes_local[i].append(np.zeros(SIZE))
                    n_nodes_halo[i].append(np.zeros(SIZE))
                    n_edges[i].append(np.zeros(SIZE))
                    for RANK in range(SIZE):

                        str_temp_1 = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_SEED_12_3_7_32_3_2_4_none.tar"
                        str_temp_2 = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_SEED_12_3_4_32_3_2_4_none.tar"
                        if os.path.exists(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}/" + str_temp_1):
                            a = torch.load(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}/" + str_temp_1)
                            #a = torch.load(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}_v2/" + str_temp_1)
                        elif os.path.exists(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}/" + str_temp_2):
                            a = torch.load(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}/" + str_temp_2)
                            #a = torch.load(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}_v2/" + str_temp_2)
                        else:
                            a = {}
                            a['n_nodes_local'] = torch.tensor(0)
                            a['n_nodes_halo'] = torch.tensor(0)
                            a['n_edges'] = 0


                        # ~~~~ old 
                        # try: 
                        #     str_temp = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_all_to_all.tar"
                        #     a = torch.load("./outputs/GraphStatistics/" + str_temp)
                        # except FileNotFoundError:
                        #     str_temp = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_2_halo_none.tar"
                        #     a = torch.load("./outputs/GraphStatistics/" + str_temp)

                        
                        n_nodes_local[i][j][RANK] = a['n_nodes_local'].item()
                        n_nodes_halo[i][j][RANK] = a['n_nodes_halo'].item()
                        n_edges[i][j][RANK] = a['n_edges']

            n_nodes_local_ele.append(n_nodes_local)
            n_nodes_halo_ele.append(n_nodes_halo)
            n_edges_ele.append(n_edges)


        # Local nodes per rank 
        ms = 100
        fig, ax = plt.subplots(figsize=(8,7))
        for e in range(len(NELE_LIST)):
            for j in range(len(SIZE_LIST)):
                SIZE = SIZE_LIST[j]
                ax.scatter(SIZE*np.ones(SIZE), n_nodes_local_ele[e][0][j], s=ms, color='black')
                ax.text(SIZE, n_nodes_local_ele[e][0][j][0], NELE_LIST[e], color='blue')
                print(f"Nele={NELE_LIST[e]}, SIZE={SIZE}, nodes={n_nodes_local_ele[e][0][j][0]}")
                #ax.scatter(SIZE*np.ones(SIZE), n_nodes_local_ele[e][1][j], s=ms, color='blue', marker=marker[e]) 
                #ax.scatter(SIZE*np.ones(SIZE), n_nodes_local_ele[e][2][j], s=ms, color='red', marker=marker[e]) 
                #ax.scatter(SIZE*np.ones(SIZE), n_nodes_local[3][j], s=ms, color='green') 
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Local Graph Nodes')
        ax.set_xlabel('Number of GPUs')
        #ax.legend(framealpha=1)
        plt.show(block=False)

        # ~~~~ # # Halo nodes per rank 
        # ~~~~ # ms = 100
        # ~~~~ # fig, ax = plt.subplots(figsize=(8,7))
        # ~~~~ # for j in range(len(SIZE_LIST)):
        # ~~~~ #     SIZE = SIZE_LIST[j]
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[0][j], s=ms, color='black') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[1][j], s=ms, color='blue') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[2][j], s=ms, color='red') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[3][j], s=ms, color='green') 
        # ~~~~ # ax.set_yscale('log')
        # ~~~~ # ax.set_xscale('log')
        # ~~~~ # ax.set_ylabel('Halo Graph Nodes')
        # ~~~~ # ax.set_xlabel('Number of GPUs')
        # ~~~~ # plt.show(block=False)


        # ~~~~ # # Halo nodes / local nodes 
        # ~~~~ # ms = 100
        # ~~~~ # fig, ax = plt.subplots(figsize=(8,7))
        # ~~~~ # for j in range(len(SIZE_LIST)):
        # ~~~~ #     SIZE = SIZE_LIST[j]
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[0][j]/n_nodes_local[0][j], s=ms, color='black') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[1][j]/n_nodes_local[1][j], s=ms, color='blue') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[2][j]/n_nodes_local[2][j], s=ms, color='red') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[3][j]/n_nodes_local[3][j], s=ms, color='green') 
        # ~~~~ # ax.set_yscale('log')
        # ~~~~ # ax.set_xscale('log')
        # ~~~~ # ax.set_ylabel('Halo Nodes / Local Nodes')
        # ~~~~ # ax.set_xlabel('Number of GPUs')
        # ~~~~ # plt.show(block=False)


        # ~~~~ # # # Total halo nodes (summed over all ranks)
        # ~~~~ # # ms = 100
        # ~~~~ # # fig, ax = plt.subplots(figsize=(8,7))
        # ~~~~ # # for j in range(len(SIZE_LIST)):
        # ~~~~ # #     SIZE = SIZE_LIST[j]
        # ~~~~ # #     ax.scatter(SIZE, np.sum(n_nodes_halo[0][j]), s=ms, color='black') 
        # ~~~~ # #     ax.scatter(SIZE, np.sum(n_nodes_halo[1][j]), s=ms, color='blue') 
        # ~~~~ # #     ax.scatter(SIZE, np.sum(n_nodes_halo[2][j]), s=ms, color='red') 
        # ~~~~ # #     ax.scatter(SIZE, np.sum(n_nodes_halo[3][j]), s=ms, color='green') 
        # ~~~~ # # ax.set_yscale('log')
        # ~~~~ # # ax.set_xscale('log')
        # ~~~~ # # ax.set_ylabel('Total Halo Graph Nodes')
        # ~~~~ # # ax.set_xlabel('Ranks')
        # ~~~~ # # plt.show(block=False)


        # ~~~~ # # Edges per rank 
        # ~~~~ # ms = 100
        # ~~~~ # fig, ax = plt.subplots(figsize=(8,7))
        # ~~~~ # for j in range(len(SIZE_LIST)):
        # ~~~~ #     SIZE = SIZE_LIST[j]
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_edges[0][j], s=ms, color='black') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_edges[1][j], s=ms, color='blue') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_edges[2][j], s=ms, color='red') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_edges[3][j], s=ms, color='green') 
        # ~~~~ # ax.set_yscale('log')
        # ~~~~ # ax.set_xscale('log')
        # ~~~~ # ax.set_ylabel('Graph Edges')
        # ~~~~ # ax.set_xlabel('Number of GPUs')
        # ~~~~ # plt.show(block=False)

    if 1 == 0:
        """
        Looking at profiler outputs 
        """
        if 1 == 0: # data generation 
            profile_path = "./outputs/profiles/weak_scale_v2_updated/"
            POLY_LIST = [3,5] # nekrs polynomial order  
            N_MP_LIST = [2,4,6,8] # number of message passing layers 
            N_HC_LIST = [8,16,32] # number of hidden channels 

            # constants 
            seed = 12
            input_channels_node = 3
            input_channels_edge = 4
            output_channels = 3
            hidden_layers = 2 

            SIZE_LIST = [1,2,4,8,16,32,64]
            #SIZE_LIST = [1,2,4,8,16,32,64,128]
            HALO_MODE_LIST = ['none', 'all_to_all', 'send_recv']

            for poly in POLY_LIST:
                for n_mp in N_MP_LIST:
                    for n_hc in N_HC_LIST:
                        t_forwardPass_cuda = {}
                        t_forwardPass_cpu = {}
                        t_indexAdd_cuda = {}
                        t_indexAdd_cpu = {}
                        for i in range(len(HALO_MODE_LIST)):
                            halo = HALO_MODE_LIST[i]
                            t_forwardPass_cuda[halo] = []
                            t_forwardPass_cpu[halo] = []
                            t_indexAdd_cuda[halo] = []
                            t_indexAdd_cpu[halo] = []
                            for j in range(len(SIZE_LIST)):
                                size = SIZE_LIST[j]
                                t_forwardPass_cuda[halo].append(np.zeros(size))
                                t_forwardPass_cpu[halo].append(np.zeros(size))
                                t_indexAdd_cuda[halo].append(np.zeros(size))
                                t_indexAdd_cpu[halo].append(np.zeros(size))
                                for k in range(size):
                                    rank = k 
                                    # file_str = f"POLY_{poly}_RANK_{rank}_SIZE_{size}_input_channels_3_hidden_channels_{n_hc}_output_channels_3_nMessagePassingLayers_{n_mp}_halo_{halo}.tar"
                                    file_str = f"POLY_{poly}_RANK_{rank}_SIZE_{size}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}.tar"

                                    # load profile data 
                                    try: 
                                        temp_prof = torch.load(profile_path + file_str)
                                        #temp_prof = torch.load("./outputs/profiles_old/" + file_str)
                                        # print(temp_prof.table(sort_by="cpu_time_total", row_limit=10))
                                        key_list = [] 
                                        for key_id in range(len(temp_prof)):
                                            key_list.append(temp_prof[key_id].key)
                                        idx_key = key_list.index(f'[RANK {rank}] FORWARD PASS') 
                                        cuda_time = temp_prof[idx_key].cuda_time # in microseconds,averaged over runs  
                                        cpu_time = temp_prof[idx_key].cpu_time # in microseconds  

                                        if 'aten::index_add_' in key_list:
                                            idx_key = key_list.index('aten::index_add_')
                                            indexAdd_cuda_time = temp_prof[idx_key].cuda_time
                                            indexAdd_cpu_time = temp_prof[idx_key].cpu_time
                                        else:
                                            indexAdd_cuda_time = 0.
                                            indexAdd_cpu_time = 0.

                                    except FileNotFoundError:
                                        print(f"FileNotFound: {file_str}")
                                        cuda_time = 0
                                        cpu_time = 0
                                        indexAdd_cuda_time = 0
                                        indexAdd_cpu_time = 0

                                    t_forwardPass_cuda[halo][j][k] = cuda_time
                                    t_forwardPass_cpu[halo][j][k] = cpu_time
                                    t_indexAdd_cuda[halo][j][k] = indexAdd_cuda_time
                                    t_indexAdd_cpu[halo][j][k] = indexAdd_cpu_time
                                    print(f"[POLY {poly}, N_MP {n_mp}, N_HC {n_hc}, SIZE {size}, RANK {rank}] -- cuda_time = {cuda_time} us, indexAdd_cuda_time = {indexAdd_cuda_time} us")

                        
                        # Write the data 
                        for halo in HALO_MODE_LIST:
                            x_axis = np.array(SIZE_LIST)
                            y_axis_mean = np.zeros_like(x_axis)
                            y_axis_max = np.zeros_like(x_axis)
                            y_axis_min = np.zeros_like(x_axis)
                            for j in range(len(SIZE_LIST)):
                                y_axis_mean[j] = t_forwardPass_cuda[halo][j].mean()
                                y_axis_max[j] = t_forwardPass_cuda[halo][j].max()
                                y_axis_min[j] = t_forwardPass_cuda[halo][j].min()
                            
                            temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                            np.save(profile_path + f"{temp_name}_mean_cuda.npy", y_axis_mean)
                            np.save(profile_path + f"{temp_name}_max_cuda.npy", y_axis_min)
                            np.save(profile_path + f"{temp_name}_min_cuda.npy", y_axis_max)

                        for halo in HALO_MODE_LIST:
                            x_axis = np.array(SIZE_LIST)
                            y_axis_mean = np.zeros_like(x_axis)
                            y_axis_max = np.zeros_like(x_axis)
                            y_axis_min = np.zeros_like(x_axis)
                            for j in range(len(SIZE_LIST)):
                                y_axis_mean[j] = t_forwardPass_cpu[halo][j].mean()
                                y_axis_max[j] = t_forwardPass_cpu[halo][j].max()
                                y_axis_min[j] = t_forwardPass_cpu[halo][j].min()
                            
                            temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                            np.save(profile_path + f"{temp_name}_mean_cpu.npy", y_axis_mean)
                            np.save(profile_path + f"{temp_name}_max_cpu.npy", y_axis_min)
                            np.save(profile_path + f"{temp_name}_min_cpu.npy", y_axis_max)

                        # Write the data -- indexAdd
                        for halo in HALO_MODE_LIST:
                            x_axis = np.array(SIZE_LIST)
                            y_axis_mean = np.zeros_like(x_axis)
                            y_axis_max = np.zeros_like(x_axis)
                            y_axis_min = np.zeros_like(x_axis)
                            for j in range(len(SIZE_LIST)):
                                y_axis_mean[j] = t_indexAdd_cuda[halo][j].mean()
                                y_axis_max[j] = t_indexAdd_cuda[halo][j].max()
                                y_axis_min[j] = t_indexAdd_cuda[halo][j].min()
                            
                            temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                            np.save(profile_path + f"{temp_name}_mean_indexAdd_cuda.npy", y_axis_mean)
                            np.save(profile_path + f"{temp_name}_max_indexAdd_cuda.npy", y_axis_min)
                            np.save(profile_path + f"{temp_name}_min_indexAdd_cuda.npy", y_axis_max)

                        for halo in HALO_MODE_LIST:
                            x_axis = np.array(SIZE_LIST)
                            y_axis_mean = np.zeros_like(x_axis)
                            y_axis_max = np.zeros_like(x_axis)
                            y_axis_min = np.zeros_like(x_axis)
                            for j in range(len(SIZE_LIST)):
                                y_axis_mean[j] = t_indexAdd_cpu[halo][j].mean()
                                y_axis_max[j] = t_indexAdd_cpu[halo][j].max()
                                y_axis_min[j] = t_indexAdd_cpu[halo][j].min()
                            
                            temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                            np.save(profile_path + f"{temp_name}_mean_indexAdd_cpu.npy", y_axis_mean)
                            np.save(profile_path + f"{temp_name}_max_indexAdd_cpu.npy", y_axis_min)
                            np.save(profile_path + f"{temp_name}_min_indexAdd_cpu.npy", y_axis_max)

        # Scaling plots 
        if 1 == 1:
            profile_path = "./outputs/profiles/weak_scale_v2_updated/"
            HALO_MODE_LIST = ['none', 'all_to_all', 'send_recv']
            seed = 12
            input_channels_node = 3
            input_channels_edge = 4
            output_channels = 3 
            hidden_layers = 2

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~ the effect of n_mp layers, for fixed poly and n_hc 
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            poly = 5
            n_hc = 8
            N_MP_LIST = [2,4,6,8]
            #N_MP_LIST = [2,6]
            data_str = 'cuda'
            #data_str = 'cpu'
            #data_str = 'indexAdd_cuda'
            #data_str = 'indexAdd_cpu'

            if poly==3:
                #n_nodes_local = 116400
                #n_nodes_local = np.array([110592, 117504, 117504, 117504, 128304, 128304, 128304]) # includes halo 
                n_nodes_local = np.array([110592, 112896, 112896, 112896, 116400, 116400, 116400]) # no halo nodes
            if poly==5:
                #n_nodes_local = 528080
                #n_nodes_local = np.array([512000, 531200, 531200, 531200, 560720, 560720, 560720]) # includes halo 
                n_nodes_local = np.array([512000, 518400, 518400, 518400, 528080, 528080, 528080]) # no halo nodes 

            norm = Normalize(vmin=np.min(N_MP_LIST), vmax=np.max(N_MP_LIST))

            data_all_mean = [] 
            data_all_max = [] 
            data_all_min = []

            data_none_mean = []
            data_none_max = []
            data_none_min = []

            data_sr_mean = []
            data_sr_max = []
            data_sr_min = []

            for n_mp in N_MP_LIST:
                halo = 'all_to_all'
                temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                data_all_mean.append(np.load(profile_path + f"{temp_name}_mean_{data_str}.npy"))
                data_all_max.append(np.load(profile_path + f"{temp_name}_max_{data_str}.npy"))
                data_all_min.append(np.load(profile_path + f"{temp_name}_min_{data_str}.npy"))

                halo = 'none'
                temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                data_none_mean.append(np.load(profile_path + f"{temp_name}_mean_{data_str}.npy"))
                data_none_max.append(np.load(profile_path + f"{temp_name}_max_{data_str}.npy"))
                data_none_min.append(np.load(profile_path + f"{temp_name}_min_{data_str}.npy"))

                halo = 'send_recv'
                temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                data_sr_mean.append(np.load(profile_path + f"{temp_name}_mean_{data_str}.npy"))
                data_sr_max.append(np.load(profile_path + f"{temp_name}_max_{data_str}.npy"))
                data_sr_min.append(np.load(profile_path + f"{temp_name}_min_{data_str}.npy"))

            #x_axis = np.array([1,2,4,8,16,32,64,128])
            x_axis = np.array([1,2,4,8,16,32,64])

            # Get total throughput: n_nodes / time 
            eff_all = [] 
            for i in range(len(N_MP_LIST)):
                t_fp = data_all_mean[i]
                total_nodes = (n_nodes_local*x_axis)
                total_throughput = total_nodes/t_fp
                serial_throughput = total_throughput[0]
                eff = total_throughput/(serial_throughput * x_axis)
                eff_all.append(eff)

            eff_sr = [] 
            for i in range(len(N_MP_LIST)):
                t_fp = data_sr_mean[i]
                total_nodes = (n_nodes_local*x_axis)
                total_throughput = total_nodes/t_fp
                serial_throughput = total_throughput[0]
                eff = total_throughput/(serial_throughput * x_axis)
                eff_sr.append(eff)

            lw = 1.5 
            ms = 14
            fig, ax = plt.subplots(1,2,figsize=(10,6), sharex=True)
            for i in range(len(N_MP_LIST)):
                color = cm.viridis(norm(N_MP_LIST[i]))
                ax[0].plot(x_axis[:-1], data_none_mean[i][:-1], color=color, lw=lw, marker='o', ms=ms, mew=1.5, fillstyle='none', mec=color, label='none')
                ax[0].plot(x_axis[:-1], data_all_mean[i][:-1],  color=color, lw=lw, marker='s', ms=ms, mew=1.5, fillstyle='none', mec=color, ls='--', label='all_to_all')
                ax[0].plot(x_axis[:-1], data_sr_mean[i][:-1],   color=color, lw=lw, marker='^', ms=ms, mew=1.5, fillstyle='none', mec=color, ls='-.', label='send_recv')
                ax[1].plot(x_axis, eff_all[i]*100,    color=color, lw=lw, marker='s', ms=ms, mew=1.5, fillstyle='none', mec=color, ls='--')
                ax[1].plot(x_axis, eff_sr[i]*100,     color=color, lw=lw, marker='^', ms=ms, mew=1.5, fillstyle='none', mec=color, ls='-.')

            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            ax[0].set_xlabel('nGPU')
            ax[0].set_ylabel(f'{data_str} time [us]')
            ax[0].set_title(f"poly={poly}, hc={n_hc}")
            ax[0].set_ylim([1e3, 1e6])
            #ax[0].legend(fancybox=False, framealpha=1)

            ax[1].set_xscale('log')
            ax[1].set_xlabel('nGPU')
            ax[1].set_ylabel('Throughput Efficiency [%]')
            ax[1].set_title(f"poly={poly}, hc={n_hc}")
            ax[1].set_ylim([0., 105])

            # # left, bottom, width, height
            # cax = fig.add_axes([0.11, 0.21, 0.35, 0.03])  # Position and size of the color bar
            # sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            # sm.set_array([])
            # fig.colorbar(sm, cax=cax, orientation='horizontal')
            # ax.grid(False)
            plt.show(block=False)

            # # ~~~~ OLD PLOTTING 
            # # Plot all curves 
            # data_all_mean = [] 
            # data_all_max = [] 
            # data_all_min = []

            # data_none_mean = []
            # data_none_max = []
            # data_none_min = []

            # for poly in [1,3,5]:
            #     halo = 'all_to_all'
            #     data_all_mean.append(np.load(f"outputs/p_{poly}_{halo}_mean.npy"))
            #     data_all_max.append(np.load(f"outputs/p_{poly}_{halo}_max.npy"))
            #     data_all_min.append(np.load(f"outputs/p_{poly}_{halo}_min.npy"))

            #     halo = 'none'
            #     data_none_mean.append(np.load(f"outputs/p_{poly}_{halo}_mean.npy"))
            #     data_none_max.append(np.load(f"outputs/p_{poly}_{halo}_max.npy"))
            #     data_none_min.append(np.load(f"outputs/p_{poly}_{halo}_min.npy"))

            # x_axis = np.array([1,2,4,8,16,32,64,128])


            # lw = 2 
            # fig, ax = plt.subplots(figsize=(6,6))

            # ax.plot(x_axis, data_none_mean[0], color='black', lw=lw, marker='o', label='p=1, no halo')
            # ax.plot(x_axis, data_all_mean[0], color='black', ls='--', lw=lw, marker='s', label='p=1, all_to_all')

            # ax.plot(x_axis, data_none_mean[1], color='blue', lw=lw, marker='o', label='p=3, no halo')
            # ax.plot(x_axis, data_all_mean[1], color='blue', ls='--', lw=lw, marker='s', label='p=3, all_to_all')

            # ax.plot(x_axis[3:], data_none_mean[2][3:], color='red', lw=lw, marker='o', label='p=5, no halo')
            # ax.plot(x_axis[3:], data_all_mean[2][3:], color='red', ls='--', lw=lw, marker='s', label='p=5, all_to_all')

            # ax.set_xscale('log')
            # ax.set_yscale('log')
            # ax.set_xlabel('nGPU')
            # ax.set_ylabel('Time [us]')
            # #ax.legend(fancybox=False, framealpha=1)
            # plt.show(block=False)


    if 1 == 0:
        """
        Test profiler output 
        """
        a = torch.load('./outputs/profiles/weak_scale/POLY_3_RANK_3_SIZE_32_SEED_12_3_4_32_3_2_4_all_to_all.tar')
        b = torch.load('./outputs/profiles/weak_scale/POLY_3_RANK_3_SIZE_32_SEED_12_3_4_32_3_2_4_none.tar')

        a_keys = [] 
        b_keys = []
        for i in range(len(a)):
            a_keys.append(a[i].key)
        for i in range(len(b)):
            b_keys.append(b[i].key)



        c = [item for item in a if item not in b]





        





