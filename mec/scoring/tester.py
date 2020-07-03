import json
import torch
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from PIL import Image

# ----- ----- ----- ----- 混淆矩阵测试 ----- ----- ----- -----

def test_mix_sync(
        model,  
        test_loader, 
        idx_to_class_list, # 类别列表，有序排列
        output_file_path,
        device=torch.device('cpu')):
    model = model.to(device)
    model.eval()
    num_classes = len(idx_to_class_list)
    mix_mat_count_tensor = torch.zeros((num_classes, num_classes+1), dtype=torch.long).to(device)
    with torch.no_grad():
        for test_data_batch, test_target_index_batch in tqdm(test_loader, ncols=80):
            test_data_batch = test_data_batch.to(device)
            test_output_batch = model(test_data_batch)
            test_output_index_batch = test_output_batch.max(dim=-1)[1]
            for test_target_index, test_output_index in zip(test_target_index_batch, test_output_index_batch):
                mix_mat_count_tensor[test_target_index][test_output_index] += 1 # 计数一次判断
                mix_mat_count_tensor[test_target_index][num_classes]       += 1 # 总数技术
    mix_mat_count_list = mix_mat_count_tensor.tolist()
    counts_dict    = { 'name': ['总计'] + idx_to_class_list }
    propotion_dict = { 'name': ['总计', '精确率', '召回率', 'F1-score'] + idx_to_class_list} # propotion dict
    for i in range(num_classes):
        #print(mix_mat_count_list)
        counts_column = mix_mat_count_list[i]
        counts_column = [counts_column[num_classes]] + counts_column[0:num_classes]
        column_total = counts_column[0]
        propotion_column = [column_total]
        row_total = torch.sum(mix_mat_count_tensor[:,i]).item()
        true_positive_count = counts_column[i+1]
        real_positive_count = column_total
        pred_positive_total = row_total
        # precision
        precision = true_positive_count/pred_positive_total if pred_positive_total>0 else 0.0
        propotion_column.append(precision)
        # recall
        recall = true_positive_count/real_positive_count if real_positive_count>0 else 0.0
        propotion_column.append(recall)
        # f1
        f1_score = 2*true_positive_count/(real_positive_count+pred_positive_total) if true_positive_count>0 else 0.0
        propotion_column.append(f1_score)
        for j in range(1, num_classes+1): # 比例
            propotion_column.append(counts_column[j]/column_total if column_total>0 else 0.0)
        # 正确标签名
        target_class_name = idx_to_class_list[i] 
        # ----- -----
        counts_dict[ target_class_name ] = counts_column 
        propotion_dict[ target_class_name ] = propotion_column # 总数
    
    count_frame     = pd.DataFrame(data=counts_dict)
    propotion_frame = pd.DataFrame(data=propotion_dict)
    writer = pd.ExcelWriter(output_file_path, engine='xlsxwriter')

    count_frame.to_excel(writer, sheet_name='判别计数', index=False)
    workbook1  = writer.book
    worksheet1 = writer.sheets['判别计数']
    worksheet1.set_column(0, 0, 8)
    worksheet1.set_column(1, num_classes, 5)

    propotion_frame.to_excel(writer, sheet_name='判别比例', index=False)
    workbook2  = writer.book
    worksheet2 = writer.sheets['判别比例']
    worksheet2.set_column(0, 0, 8)
    worksheet2.set_column(1, num_classes, 5)

    writer.save()
    return 

def test_mix_async(
        model, 
        test_set,
        train_idx_to_class = None, 
        test_idx_to_class  = None,
        test_loader=None):
    num_classes = len(idx_to_class_list) 
    mix_mat_count_tensor = torch.zeros((num_classes, num_classes+1), dtype=torch.long).to(device)
    
    pass

def test_mix(
        model, 
        test_set = None,
        train_idx_to_class = None,
        test_idx_to_class  = None,
        test_loader=None):
    """ 测试model在test_set上的准确率及混淆矩阵
        * 要求单标签流程
        * test_set跟model的序号排列不同时，要求给出训练集和测试集上的两个idx_to_class
        参数：
            model               : 测试用的模型
            test_set            : 测试针对的数据集
            test_loader         : 测试数据集的读取器，不可与test_set共存
            train_idx_to_class  : 训练集的类别排列，indexable
            test_idx_to_class   : 测试集的类别排列
    """
