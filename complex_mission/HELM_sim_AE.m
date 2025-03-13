function output=HELM_sim_AE(para,train_x,train_y,test_x,test_y)
   

% Usage: output = HELM(para,train_x,train_y,test_x,test_y)
%
% Input:
% para: structure variable
% para.layer_neu: number of neurals in each layer, a vector, the last layer is ELM, rests are autoencoder 
% para.C: C parameter in ELM layer
% para.weight_norm: norm weight parameter in autoencoder layer, with l1 norm
 %para.Elm_Type: 0 for regression; 1 for (both binary and multi-classes) classification
% train_x:  training data set, row with patterns, and column with features
% train_y:  for the case of unsupervised learning, set train_y=0, otherwise, label of training set (for classification) or corresponding values for training set (regression)
% test_x: testing data set, row with patterns, and column with features
% test_y (opinional): label of testing set (for classification) or corresponding values for testing  set (regression)


% Output: 
% output.TrainingTime                - Time (seconds) spent on training ELM
% output.TestingTime                 - Time (seconds) spent on predicting ALL testing data
% output.TrainingAccuracy            - Training accuracy: 
%                               RMSE for regression or correct classification rate for classification
% output.TestingAccuracy             - Testing accuracy: 
%                               RMSE for regression or correct classification rate for classification
% output.train_fea: the features that learned by HELM in each layer in
% training set
% output.test_fea: the features that learned by HELM in each layer in
% testing set
% output.train_weight: the weight of each layer in HELM (auto-encoder part)
% output.beta_last: the weight of the last supervised layer in HELM

        output.para=para;
        para.layer_num=length(para.layer_neu);
        REGRESSION=0;
        CLASSIFIER=1;
        stand_factor=[];
        if nargin>2
            true=train_y;
        end
        if para.Elm_Type~=REGRESSION
            %%%%%%%%%%%% Preprocessing the data of classification
            %%%%%%%%%% Processing the targets of training
%             temp_T=ind2vec(train_y');
%             train_y=temp_T*2-1;
%             train_y=train_y';
            
            
        end
        tic
        % train_x = zscore(train_x);
        [input, raw_ps]= mapminmax(train_x',0,1);
        input=input';
        output.train_fea{1}=input;
%         input = mapminmax(train_x',0,1)';
   
        %%%%%%%%%%%% layer 1 to n-1: autoencoder with l1 norm %%%%%%%%%%%%%
        
        for i=1:para.layer_num-1 % from layer 1 to layer n-1 is the autoencoder for feature learning, and the layer n is the ELM

                H = [input .1 * ones(size(input,1),1)];
                b=2*rand(size(H,2),para.layer_neu(i))-1;
                A = H * b;
                A = tansig(A);
                A = mapminmax(A')';
                clear b;
                beta  =  sparse_elm_autoencoder(A,H,para.weight_norm,50)';
                clear A;
                T=angle_cal_mex(H,beta);
                T_temp = T* beta';
                err=T_temp-H;
                if para.isprint==1
                    fprintf(1,['Layer' num2str(i) ': Max error of Output %f Min Val %f\n'],max(err(:)),min(err(:)));
                end
                
                
                clear H;
                output.train_fea{i}=T;
                output.train_weight{i}=beta;
                input=T;

        end
      %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%%   
      output.beta_last =[];
      output.raw_ps=raw_ps;
      if nargin>2
    %%%%%%%% last layer ELM %%%%%%%%%%%%%%
            H = [input .1 * ones(size(input,1),1)];
            if length(para.layer_neu)==1
                b=orth(2*rand(size(H,2),para.layer_neu(1))'-1)';
            else
                b=orth(2*rand(size(H,2),para.layer_neu(i+1))'-1)';
            end
            A = H * b;
            A = tansig(A);
            output.b =b;
            output.beta_last = (A'  * A+eye(size(A',1)) * (para.C)) \ ( A'  *  train_y);
            Training_time = toc;
            output.TrainingTime=Training_time;
            if para.isprint==1
                disp('Training has been finished!');
                disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
            end


            % %% Calculate the training accuracy
            train_output = A * output.beta_last;
            output.train_output=train_output;
            if para.Elm_Type~=REGRESSION
                zz=vec2ind(train_output')';
                temp=abs(zz-true);
                wrong_class=length(find(temp>0));
                output.trainingAccuracy=1-wrong_class/size(train_output,1)  ;
                if para.isprint==1
                      disp(['Training Accuracy is : ', num2str(output.trainingAccuracy )]);
                end
            else
                output.trainingAccuracy=sqrt(mse(train_output- true));
                if para.isprint==1
                    disp(['Training Accuracy is : ', num2str(output.trainingAccuracy ) ]);
                end
            end
      end
      
      %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%%   
        output.stand_factor=stand_factor;
      
         %%%%%%%%%%%% test set %%%%%%%%%%%%%
     if nargin>3
         tic
         input  =  mapminmax('apply',test_x',raw_ps)';
%             input = mapminmax(test_x',0,1)';
        output.test_fea{1}=input;
         for i=1:para.layer_num-1 % from layer 1 to layer n-1 is the autoencoder for calculate the feature

                H = [input .1 * ones(size(input,1),1)];
                T=angle_cal_mex(H,output.train_weight{i});
                
                output.test_fea{i}=T;
                input=T;
                
         end
        
         if size(true,2)~=0
                H = [input .1 * ones(size(input,1),1)];
                A = H * b;
                A = tansig(A);
                output.test_output = A * output.beta_last;
                Testing_time = toc;
                if para.isprint==1
                    disp('Testing set calculation has been finished!');
                    disp(['The Total calculation Time is : ', num2str(Testing_time), ' seconds' ]);
                end
                output.Testing_time=Testing_time;     
         end
        output.b=b;
        if nargin==5
        % %% Calculate the testing accuracy
                if para.Elm_Type~=REGRESSION
                    zz=vec2ind(output.test_output')';
                    temp=abs(zz-test_y);
                    wrong_class=find(temp>0); 
                    output.right_class=find(temp==0);
                    output.testingAccuracy=1-length(wrong_class)/size(output.test_output,1)  ;
                    output.test_label=zz;
                    output.wrong_class=wrong_class;
                    if para.isprint==1
                        disp(['Testing Accuracy is : ', num2str(output.testingAccuracy ) ]);
                    end
                else
                    output.testingAccuracy=sqrt(mse(output.test_output- test_y));
                    if para.isprint==1
                        disp(['Testing Accuracy is : ', num2str(output.testingAccuracy ) ]);
                    end
                end
        end
     end
end
        
      %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%%   

