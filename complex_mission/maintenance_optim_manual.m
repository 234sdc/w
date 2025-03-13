clc
clear all
close all

N=20;
ini_health=20;
nums=10;
thre=1000;% totle times for a epoch
record_input=zeros(thre*N,62);
record_action=zeros(thre*N,3);
counter=1;




for i=1:N
    t=1;
    
    states=mission_profile(nums);
    health_true=ini_health;

    input_set=[states.length  states.mission_timepoint  states.mission_strengh  ...
            states.mission_reward  states.maintenance_time  states.maintenance_cost];
    temp=[t health_true reshape(input_set',1,[])];


    
    input1(1,1:62)=temp;
    reward=0;
    cost=0;
    while t<thre
        [reward_temp,cost_temp,input2,action]=manual_action(input1,ini_health);
        record_input(counter,:)=input1;
        record_action(counter,:)=action;
        t=input2(1);
        reward=reward+reward_temp;
        cost=cost+cost_temp;
        input1=input2;
        counter=counter+1;
    end
    score(i)=reward-cost;
end
record_input(counter+1:end,:)=[];
record_action(counter+1:end,:)=[];
figure
subplot(211)
plot(score);
subplot(212)
hist(score,30);
mean(score)
std(score)

        
        


