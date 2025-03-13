clc
clear all
close all

N=500;
ini_health=40;
nums=20;
num_mission=26001;
thre=50;% totle times for a epoch
dur=1200;
counter=0;
co1=0;
T=0.85*ini_health; % schedule durision of schedule maintenance

record_input=zeros(thre*N,nums*8+3);
record_action=zeros(thre*N,6);
extre_mission=0.1;

action_sta_elm=zeros(N*thre,6);
action_sta_manu=zeros(N*thre,6);
action_sta_after=zeros(N*thre,6);
action_sta_sche=zeros(N*thre,6);

results_manu=zeros(thre*N,5);
results_elm=zeros(thre*N,5);
results_after=zeros(thre*N,5);
results_sche=zeros(thre*N,5);

epsilon=0.05;
learn_rate=0.99;
PHM_noise=0.05;



para.layer_neu=[ 500 ];
para.C=1e-5;
para.weight_norm=1e-5;
para.Elm_Type=1; % 0 for regression; 1 for (both binary and multi-classes) classification
para.isprint=0;
zeta=0;
retrain_point=[];


states_vector=mission_profile_all(num_mission,extre_mission);
spare_store=0;
temp=[0 ini_health spare_store reshape(states_vector(1:nums,:)',1,[]); ];
states_vector(1:nums,:)=[];


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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    input1_elm=input1(1,:);
    input1_manu=input1(1,:);
    input1_after=input1(1,:);
    input1_sche=input1(1,:);
    
    reward_elm=0;
    cost_elm=0;
    reward_manu=0;
    cost_manu=0;
    reward_after=0;
    cost_after=0;
    reward_sche=0;
    cost_sche=0;
    
    spare_queue_manu=[];
    spare_queue_elm=[];
    spare_queue_after=[];
    spare_queue_sche=[];
    
    timer=0; % counter for schedule maintenance
    
    while t<=thre
    co1=co1+1;
    counter=counter+1;
    
    %%%%%%%%%%%%%%%%% after event section %%%%%%%%%%%%%%%%%%%
        [reward_temp,cost_temp,cost_store,input2_after,action_after,spare_queue_after]=after_event_action(input1_after,...,
            ini_health,spare_queue_after,nums,states_vector(co1,:));
        action_sta_after(co1,:)=action_after;
        reward_after=reward_after+reward_temp;
        cost_after=cost_after+cost_temp;
        results_after(co1,:)=[reward_temp cost_store cost_temp-cost_store input1_after(7) input1_after(1)];
        input1_after=input2_after;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    
    
  
  %%%%%%%%%%%%%%%%% schedule event section %%%%%%%%%%%%%%%%%%%
        [reward_temp,cost_temp,cost_store,input2_sche,action_sche,spare_queue_sche,timer]=schedule_action(input1_sche,...,
            ini_health,spare_queue_sche,nums,states_vector(co1,:),timer,T);
        action_sta_sche(co1,:)=action_sche;
        reward_sche=reward_sche+reward_temp;
        cost_sche=cost_sche+cost_temp;
        results_sche(co1,:)=[reward_temp cost_store cost_temp-cost_store input1_sche(7) input1_sche(1)];
        input1_sche=input2_sche;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    
    
    
    %%%%%%%%%%%%%%%%% PHM manu section %%%%%%%%%%%%%%%%%%%
        [reward_temp,cost_temp,cost_store,input2_manu,action_manu,spare_queue_manu]=manual_action(input1_manu,...,
            ini_health,spare_queue_manu,nums,states_vector(co1,:),PHM_noise);
        action_sta_manu(co1,:)=action_manu;
        temp_manu_input=input1_manu;
        reward_manu=reward_manu+reward_temp;
        cost_manu=cost_manu+cost_temp;
        input1_manu=input2_manu;
        results_manu(co1,:)=[reward_temp cost_store cost_temp-cost_store temp_manu_input(7) temp_manu_input(1)];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
        
        
     %%%%%%%%%%%%%%%%% ELM reinforcement learning section %%%%%%%%%%%%%%%%%%%
        input1_elm(2)=input1_elm(2)+input1_elm(2)*PHM_noise*randn;
        if rand<=epsilon
            action=rand(1,6);
        else
            action=HELM_sim_AE_apply(annmodel,input1_elm);
            action=action.test_output;
        end
        action_sta_elm(co1,:)=action;
        [reward_temp,cost_temp,cost_store,input2_elm,spare_queue_elm]=reward_cal(input1_elm,...,
            action,ini_health,spare_queue_elm,nums,states_vector(co1,:));
        reward_elm=reward_elm+reward_temp;
        cost_elm=cost_elm+cost_temp;
        results_elm(co1,:)=[reward_temp cost_store cost_temp-cost_store input1_elm(7) input1_elm(1)];
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

        % record the training data
%             record_input(counter,:)=temp_manu_input;
%             record_action(counter,:)=action_manu;
            record_input(counter,:)=input1_elm;
            record_action(counter,:)=action;
            
     
            
            if mod(counter,dur)==0 % every "dur" record, retrain the model
                counter
                annmodel=HELM_sim_AE(para,record_input,record_action);
%                 [q1,~]=size(record_action);
%                 z=randperm(q1);
%                 x=record_input(z(1:dur-1),:);
%                 y=record_action(z(1:dur-1),:);
%                 annmodel=HELM_sim_AE(para,x,y);
                retrain_point=[retrain_point i];
                counter=0;
            end
            input1_elm=input2_elm;
           t=t+1; 

    end
    score_manu(i)=reward_manu-cost_manu;
    score_elm(i)=reward_elm-cost_elm;
    score_after(i)=reward_after-cost_after;
    score_sche(i)=reward_sche-cost_sche;
    ind_(i)=co1;
end

figure
% subplot(211)
h1=plot(score_manu);
hold on
h2=plot(score_elm,'.-');
h3=plot(score_after,'-*');
h4=plot(score_sche,'-<');
h5=plot(retrain_point,0,'o');
legend([h1 h2 h3 h4],'manu','elm','after event','schedule')
% subplot(212)
% hist(score_manu,30);
% mean(score_manu(15:end))
% std(score_manu(15:end))

action_sta_elm(co1+1:end,:)=[];
action_sta_manu(co1+1:end,:)=[];
action_sta_after(co1+1:end,:)=[];
action_sta_sche(co1+1:end,:)=[];


[~,a_elm]=max(action_sta_elm,[],2);
[~,a_manu]=max(action_sta_manu,[],2);
[~,a_after]=max(action_sta_after,[],2);
[~,a_sche]=max(action_sta_sche,[],2);
% figure
% plot(a_elm-a_manu,'.');

[~,zz]=max(score_elm);
result_plot(zz,a_elm,ind_,results_elm);
result_plot(zz,a_manu,ind_,results_manu);
result_plot(zz,a_after,ind_,results_after);
result_plot(zz,a_sche,ind_,results_sche);


        
        


