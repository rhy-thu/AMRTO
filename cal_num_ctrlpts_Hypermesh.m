clc; close all; clear all;
disp(cal_num_ctrlpts('data_Hypermesh/Neovius_Hypermesh.json'));

function num_ctrlpts = cal_num_ctrlpts(file_name)
    if ~exist(file_name,'file')   
        disp(['Warning!!! File ', file_name, ' does not exist!']);
    else
        fidin=fopen(file_name); 
        [judge_begin,num_ctrlpts] = deal(0);
        while ~feof(fidin)                                                    
            tline=fgetl(fidin);                             
            if contains(tline,'"weights" :')
                judge_begin = 1;
            end
            if judge_begin == 1     
                num_ctrlpts = num_ctrlpts + 1;
                if contains(tline,']')
                    judge_begin = 0;
                    num_ctrlpts = num_ctrlpts - 3;
                end
            end
        end
    end
end