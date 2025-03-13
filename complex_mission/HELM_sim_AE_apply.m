function output=HELM_sim_AE_apply(model,test_x)
   

         para=model.para;
         para.layer_num=length(para.layer_neu);
         tic
         input  =  mapminmax('apply',test_x',model.raw_ps)';
         b=model.b;

         for i=1:para.layer_num-1 % from layer 1 to layer n-1 is the autoencoder for calculate the feature

                H = [input .1 * ones(size(input,1),1)];
                T=angle_cal_mex(H,model.train_weight{i});
                output.test_fea{i}=T;
                input=T;
                
         end
         
         if ~isempty(model.beta_last)
        
                H = [input .1 * ones(size(input,1),1)];
                A = H * b;
                A = tansig(A);
                output.test_output = A * model.beta_last;
                Testing_time = toc;
                if para.isprint==1
                    disp('Testing set calculation has been finished!');
                    disp(['The Total calculation Time is : ', num2str(Testing_time), ' seconds' ]);
                end
                output.Testing_time=Testing_time;     
         end


end
        
      %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%% %%%%%%%%   

