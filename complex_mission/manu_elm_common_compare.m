clc
clear all
close all

N=300;
ini_health=20;
nums=2;
num_mission=12000;
thre=200;% totle times for a epoch
dur=200;
counter=1;
co1=1;
co2=1;
record_input=zeros(dur,nums*8+3);
record_action=zeros(dur,6);
extre_mission=0.2;

action_sta_elm=zeros(N*thre,6);
action_sta_manu=zeros(N*thre,6);


epsilon=0.1;
learn_rate=0.95;


para.layer_neu=[ 100 ];
para.C=1e-6;
para.weight_norm=1e-5;
para.Elm_Type=1; % 0 for regression; 1 for (both binary and multi-classes) classification
para.isprint=0;
zeta=0;
retrain_point=[];

t=1;

states_vector=mission_profile_all(num_mission,extre_mission);
health_true=ini_health;
health_sense=ini_health+ini_health/10*randn;
spare_store=0;
temp=[t health_sense spare_store reshape(states_vector(1:nums,:)',1,[]); ];



%%%%%%%%%%%%% with ELM as the estimator %%%%%%%%%%%%%%%%%
input1(1,1:8*nums+3)=temp;
input1(2,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
input1(3,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
input1(4,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
input1(5,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
input1(6,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
target=diag([1 1 1 1 1 1]);
annmodel=HELM_sim_AE(para,input1,target);



for i=1:N
    t=1;
    states_vector=mission_profile_all(num_mission,extre_mission);
    health_true=ini_health;
    health_sense=ini_health+ini_health/10*randn;
    spare_store=0;
    temp=[t health_sense spare_store reshape(states_vector(1:nums,:)',1,[]); ];



    %%%%%%%%%%%%% with ELM as the estimator %%%%%%%%%%%%%%%%%
    input1(1,1:8*nums+3)=temp;
    input1(2,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
    input1(3,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
    input1(4,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
    input1(5,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
    input1(6,1:8*nums+3)=temp+1.9*randn(1,8*nums+3);
%     target=diag([1 1 1 1 1 1]);
%     annmodel=HELM_sim_AE(para,input1,target);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    input1_elm=input1(1,:);
    input1_manu=input1(1,:);
    reward_elm=0;
    cost_elm=0;
    reward_manu=0;
    cost_manu=0;
    spare_queue=[];
    %%%%%%%%%%%%%%%%% manu section %%%%%%%%%%%%%%%%%%%
    while t<thre
        [reward_temp,cost_temp,input2_manu,action_manu,spare_queue]=manual_action(input1_manu,ini_health,spare_queue,nums);
        action_sta_manu(co1,:)=action_manu;
        co1=co1+1;
        reward_manu=reward_manu+reward_temp;
        cost_manu=cost_manu+cost_temp;
        t=input2_manu(1);
        input1_manu=input2_manu;
    end
    score_manu(i)=reward_manu-cost_manu;
    
    
    
     %%%%%%%%%%%%%%%%% ELM reinforcement learning section %%%%%%%%%%%%%%%%%%%

    spare_queue=[];
    t=1;
    while t<thre 
        if rand<=epsilon
            action=rand(6,1);
        else
            action=HELM_sim_AE_apply(annmodel,input1_elm);
            action=action.test_output;
        end
        action_sta_elm(co2,:)=action;
        [reward_temp,cost_temp,input2_elm,spare_queue]=reward_cal(input1_elm,action,ini_health,spare_queue,nums);
        reward_elm=reward_elm+reward_temp;
        cost_elm=cost_elm+cost_temp;
        co2=co2+1;
        t=input2_elm(1);
%         

        
        % record the training data
        
            % Q learning part
            [~,ind]=max(action);
            action2=HELM_sim_AE_apply(annmodel,input2_elm);
            action2=action2.test_output;
            Q_sa=max(action2);
            if t>=thre
                action(ind)=reward_temp-cost_temp;
            else
                action(ind)=Q_sa*learn_rate+reward_temp-cost_temp;
%                   action(ind)=Q_sa*learn_rate+reward_elm-cost_elm;
%                     action(ind)=Q_sa*learn_rate;
%                 target(ind)=Q_sa*learn_rate;
            end

            record_input(counter,:)=input1_elm;
            record_action(counter,:)=action;
%             record_action(counter,:)=action_manu;
            counter=counter+1;
     
            
            if mod(counter,dur)==0 % every "dur" record, retrain the model
                counter
                

%                 annmodel=HELM_sim_AE(para,record_input,record_action);
                [q1,~]=size(record_action);
                z=randperm(q1);
                x=record_input(z(1:dur-1),:);
                y=record_action(z(1:dur-1),:);
                annmodel=HELM_sim_AE(para,x,y);
                retrain_point=[retrain_point i];
            end

           % only train the ELM for one time
%             if mod(counter,dur)==0&&zeta==0 % every "dur" record, retrain the model
% %                 x=record_input(1:counter-1,:);
% %                 y=record_action(1:counter-1,:);
%                 annmodel=HELM_sim_AE(para,record_input,record_action);
%                 zeta=1;
%             end
            
            
%             
            input1_elm=input2_elm;
    end
    score_elm(i)=reward_elm-cost_elm;
    ind_manu(i)=co1;
    ind_elm(i)=co2;
end

figure
% subplot(211)
h1=plot(score_manu);
hold on
h2=plot(score_elm);
h3=plot(retrain_point,0,'o');
legend([h1 h2],'manu','elm')
% subplot(212)
% hist(score_manu,30);
% mean(score_manu(15:end))
% std(score_manu(15:end))

action_sta_elm(co2+1:end,:)=[];
action_sta_manu(co1+1:end,:)=[];

[~,a_elm]=max(action_sta_elm,[],2);
[~,a_manu]=max(action_sta_manu,[],2);
% figure
% plot(a_elm-a_manu,'.');


        
        


