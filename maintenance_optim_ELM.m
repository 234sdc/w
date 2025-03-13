clc
clear all
close all

N=20;
ini_health=20;
nums=10;
thre=1000;% totle times for a epoch
dur=500;
counter=1;

record_input=zeros(dur,62);
record_action=zeros(dur,3);
epsilon=0.1;
learn_rate=.2;


para.layer_neu=[  100 ];
para.C=1e-6;
para.weight_norm=1e-5;
para.Elm_Type=1; % 0 for regression; 1 for (both binary and multi-classes) classification
para.isprint=0;
zeta=0;

t=1;

states=mission_profile(nums);
health_true=ini_health;
health_sense=ini_health+ini_health/10*randn;
input_set=[states.length  states.mission_timepoint  states.mission_strengh  ...
        states.mission_reward  states.maintenance_time  states.maintenance_cost];
temp=[t health_sense reshape(input_set',1,[])];



%%%%%%%%%%%%% with ELM as the estimator %%%%%%%%%%%%%%%%%
input1(1,1:62)=temp;
input1(2,1:62)=temp+0.9*randn(1,62);
input1(3,1:62)=temp+0.9*randn(1,62);
target=[1 0 0; 0 1 0;0 0 1];
annmodel=HELM_sim_AE(para,input1,target);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   



for i=1:N
    t=1;
    
    states=mission_profile(nums);
    health_true=ini_health;
    health_sense=ini_health+ini_health/10*randn;
    input_set=[states.length  states.mission_timepoint  states.mission_strengh  ...
            states.mission_reward  states.maintenance_time  states.maintenance_cost];
    temp=[t health_sense reshape(input_set',1,[])];


    
%%%%%%%%%%%%% with ELM as the estimator %%%%%%%%%%%%%%%%%
    input1(1,1:62)=temp;
    input1(2,1:62)=temp+0.9*randn(1,62);
    input1(3,1:62)=temp+0.9*randn(1,62);
%     target=[1 0 0; 0 1 0;0 0 1];
%     annmodel=HELM_sim_AE(para,input1,target);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    input1=input1(1,:);
    reward=0;
    cost=0;
    while t<thre
        if rand<=epsilon
            action=rand(3,1);
        else
            action=HELM_sim_AE_apply(annmodel,input1);
            action=action.test_output;
        end
        
        [reward_temp,cost_temp,input2]=reward_cal(input1,action,ini_health);
        t=input2(1);
        reward=reward+reward_temp;
        cost=cost+cost_temp;
        % record the training data
        
            % Q learning part
            [~,ind]=max(action);
            action2=HELM_sim_AE_apply(annmodel,input2);
            action2=action2.test_output;
            
            Q_sa=max(action2);
            if t>=thre
                action(ind)=reward_temp-cost_temp;
            else
                action(ind)=Q_sa*learn_rate+reward_temp-cost_temp;
%                 target(ind)=Q_sa*learn_rate;
            end
            if counter>=dur
                counter=1;
            end
            record_input(counter,:)=input1;
            record_action(counter,:)=action;
            counter=counter+1;
     
            
            if mod(counter,dur)==0 % every "dur" record, retrain the model
%                 x=record_input(1:counter-1,:);
%                 y=record_action(1:counter-1,:);
                annmodel=HELM_sim_AE(para,record_input,record_action);
            end

           % only train the ELM for one time
%             if mod(counter,dur)==0&&zeta==0 % every "dur" record, retrain the model
% %                 x=record_input(1:counter-1,:);
% %                 y=record_action(1:counter-1,:);
%                 annmodel=HELM_sim_AE(para,record_input,record_action);
%                 zeta=1;
%             end
            
            
            input1=input2;
    end
    score(i)=reward-cost;
end

figure
subplot(211)
plot(score);
subplot(212)
hist(score,30);
mean(score)
std(score)

        
        


