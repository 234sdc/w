clc
clear all
close all

N=100;
ini_health=20;
nums=10;
thre=100;% totle times for a epoch
dur=10;
counter=1;
record_max=500;
record_input=zeros(record_max,62);
record_action=zeros(record_max,3);
epsilon=0.05;
learn_rate=0.1;
layers = [
    imageInputLayer([1,nums*6+2])
    
    batchNormalizationLayer
    reluLayer   
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MaxEpochs',6, ...
    'ValidationFrequency',30,...
    'Verbose',false    );


para.layer_neu=[ 30 20 ];
para.C=1e-6;
para.weight_norm=1e-8;
para.Elm_Type=1; % 0 for regression; 1 for (both binary and multi-classes) classification
para.isprint=1;



for i=1:N
    t=1;
    
    states=mission_profile(nums);
    health_true=ini_health;
    health_sense=ini_health+ini_health/10*randn;
    input_set=[states.length  states.mission_timepoint  states.mission_strengh  ...
            states.mission_reward  states.maintenance_time  states.maintenance_cost];
    temp=[t health_sense reshape(input_set',1,[])];
%     input1=[input1;input1+0.9*randn(1,62);input1+0.9*randn(1,62)];


% %%%%%%%%%%%%% with ANN as the estimator %%%%%%%%%%%%%%%%%
%     input1(1,1:62,1,1)=temp;
%     input1(1,1:62,1,2)=temp+0.9*randn(1,62);
%     input1(1,1:62,1,3)=temp+0.9*randn(1,62);
%     target=[1;2;3];
%     target=categorical(target);
%     annmodel = trainNetwork(input1,target,layers,options);
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
%%%%%%%%%%%%% with ELM as the estimator %%%%%%%%%%%%%%%%%
    input1(1,1:62)=temp;
    input1(2,1:62)=temp+0.9*randn(1,62);
    input1(3,1:62)=temp+0.9*randn(1,62);
    target=[1 2 3];
    annmodel=HELM_sim_AE(para,input1,target);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
    reward=0;
    cost=0;
    input1=input1(:,:,:,1);
    while t<thre
        if rand<=epsilon
            action=rand(1,3);
        else
%             action = predict(annmodel,input1); % use ANN
            action = predict(annmodel,input1); % use ANN
        end
        
        [reward_temp,cost_temp,input2]=reward_cal(input1,action,ini_health);
        t=input2(1);
        reward=reward+reward_temp;
        cost=cost+cost_temp;
        % record the training data
        
            % Q learning part
            [~,ind]=max(action);
            action2 = predict(annmodel,input2);
            Q_sa=max(action2);
            if t>=thre
                action(ind)=reward-cost;
            else
                action(ind)=Q_sa*learn_rate+reward-cost;
    %             target(ind)=Q_sa*learn_rate;
            end
            if counter>=record_max
                counter=1;
            end
            record_input(counter,:)=input1;
            record_action(counter,:)=action;
            counter=counter+1;
            
            if mod(counter,dur)==0 % every "dur" record, retrain the model
                x=record_input(1:counter,:);
                y=record_action(1:counter,:);
                
                y=categorical(y); % 尝试ELM来代替神经网络
                annmodel = trainNetwork(record_input,record_action,layers,options);
            end
            input1=input2;
    end
    score=reward-cost
end

        
        


