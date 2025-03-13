function [reward_temp,cost_temp,cost_store,next,spare_queue]=reward_cal(input,act,ini_health,spare_queue,num,states_vector)
    [~,action]=max(act);
    order_t=10;
    store_cost_rate=2;
    t=input(1);
    health=input(2);
    length=input(4);
    mission_timepoint=input(5)-length;
    mission_endpoint=input(5);
    mission_strengh=input(6);
    mission_reward=input(7);
    corr_m_cost=input(8);
    corr_m_time=input(9);
    prev_m_cost=input(10);
    prev_m_time=input(11);
    spare_store=sum(spare_queue<=t);% calculate how many spare part are available at current time
    if isempty(spare_store)
        spare_store=0;
    end

    switch action
        case 1 % take the mission and order
            resu=health-length*mission_strengh;
            if (t<=mission_timepoint) % mission is acutally taken
                health=resu;
                if (resu<=0)
                    reward_temp=-3*mission_reward;
                else
                    reward_temp=mission_reward;
                end
            else
                reward_temp=0;
            end
            spare_queue=[spare_queue t+order_t];% put the new order spare part into spare queue
                        store_days=mission_endpoint-spare_queue;
            store_days(store_days>length)=length; % the maximum time of store at this term
            store_days(store_days<0)=0;% calculate the time of store 
            spare_store=sum(spare_queue<=t);% calculate current store numbers
            cost_store=store_cost_rate*sum(store_days);
            cost_temp=cost_store;
            t=mission_endpoint;
        case 2 % take the mission and don't order
            resu=health-length*mission_strengh;
            if (t<=mission_timepoint) % mission is acutally taken
                health=resu;
                if (resu<=0)
                    reward_temp=-3*mission_reward;
                else
                    reward_temp=mission_reward;
                end
            else
                reward_temp=0;
            end
            store_days=mission_endpoint-spare_queue;
            store_days(store_days>length)=length; % the maximum time of store at this term
            store_days(store_days<0)=0;% calculate the time of store 
            cost_store=store_cost_rate*sum(store_days);
            cost_temp=cost_store;
            t=mission_endpoint;
        case 3 % wait and order
            spare_queue=[spare_queue t+order_t];% put the new order spare part into spare queue
            store_days=mission_endpoint-spare_queue;
            store_days(store_days>length)=length; % the maximum time of store at this term
            store_days(store_days<0)=0;% calculate the time of store 
            spare_store=sum(spare_queue<=t);% calculate current store numbers
            cost_store=store_cost_rate*sum(store_days);
            cost_temp=cost_store;
            t=mission_endpoint;
            reward_temp=0;
            
        case 4 %     wait and don't order
            store_days=mission_endpoint-spare_queue;
            store_days(store_days>length)=length; % the maximum time of store at this term
            store_days(store_days<0)=0;% calculate the time of store 
            cost_store=store_cost_rate*sum(store_days);
            cost_temp=cost_store;
            t=mission_endpoint;
            reward_temp=0;
            
        case 5 % maintenance and order
            reward_temp=0;
            spare_queue=[spare_queue t+order_t];% put the new order spare part into spare queue
            if health<0&&spare_store>0 % able to perform the corrective maintenance
                cost_temp=corr_m_cost;
                t=t+corr_m_time;
                spare_queue(1)=[];% remove the oldest order spare part from spare queue
                health=ini_health;
                store_days=mission_endpoint-spare_queue;
                store_days(store_days>length)=length; % the maximum time of store at this term
                store_days(store_days<0)=0;% calculate the time of store 
                spare_store=sum(spare_queue<=t);% calculate current store numbers
                cost_store=store_cost_rate*sum(store_days);
                cost_temp=cost_temp+cost_store;
            else
                if health>=0&&spare_store>0 % able to perform the preventive maintenance
                    cost_temp=prev_m_cost;
                    t=t+prev_m_time;
                    spare_queue(1)=[];% remove the oldest order spare part from spare queue
                    health=ini_health;
                    store_days=mission_endpoint-spare_queue;
                    store_days(store_days>length)=length; % the maximum time of store at this term
                    store_days(store_days<0)=0;% calculate the time of store 
                    spare_store=sum(spare_queue<=t);% calculate current store numbers
                    cost_store=store_cost_rate*sum(store_days);
                    cost_temp=cost_temp+cost_store;
                else % unable to perform maintenance
                    t=mission_endpoint;
                    store_days=mission_endpoint-spare_queue;
                    store_days(store_days>length)=length; % the maximum time of store at this term
                    store_days(store_days<0)=0;% calculate the time of store 
                    cost_store=store_cost_rate*sum(store_days);
                    cost_temp=cost_store;
                end
            end
        case 6 % maintenance and don't order
            reward_temp=0;
            if health<0&&spare_store>0 % able to perform the corrective maintenance
                cost_temp=corr_m_cost;
                t=t+corr_m_time;
                spare_queue(1)=[];% remove the oldest order spare part from spare queue
                health=ini_health;
                store_days=mission_endpoint-spare_queue;
                store_days(store_days>length)=length; % the maximum time of store at this term
                store_days(store_days<0)=0;% calculate the time of store 
                spare_store=sum(spare_queue<=t);% calculate current store numbers
                cost_store=store_cost_rate*sum(store_days);
                cost_temp=cost_temp+cost_store;
            else
                if health>=0&&spare_store>0 % able to perform the preventive maintenance
                    cost_temp=prev_m_cost;
                    t=t+prev_m_time;
                    spare_queue(1)=[];% remove the oldest order spare part from spare queue
                    health=ini_health;
                    store_days=mission_endpoint-spare_queue;
                    store_days(store_days>length)=length; % the maximum time of store at this term
                    store_days(store_days<0)=0;% calculate the time of store
                    spare_store=sum(spare_queue<=t);% calculate current store numbers
                    cost_store=store_cost_rate*sum(store_days);
                    cost_temp=cost_temp+cost_store;
                else % unable to perform maintenance
                    t=mission_endpoint;
                    store_days=mission_endpoint-spare_queue;
                    store_days(store_days>length)=length; % the maximum time of store at this term
                    store_days(store_days<0)=0;% calculate the time of store 
                    cost_store=store_cost_rate*sum(store_days);
                    cost_temp=cost_store;
                end
            end
    end

   % next state
        temp=[t health spare_store];
        input(1:3)=[];% remove the health state, t and spare_store, then reshape input as the num*9
        input=reshape(input,[8,num]);    
        input=input';   
        input(1:end-1,:)=input(2:end,:);% remove the state data of current time
        input(end,:)=states_vector;% add the newest state data at the end
        input(end,2)=input(end,1)+input(end-1,2); % calculate the mission_timepoint of the newest state
        next=[temp reshape(input',1,[])];

        

end