function [reward_temp,cost_temp,cost_store,next,action,spare_queue,timer]=schedule_action(input,ini_health,spare_queue,num,states_vector,timer,T)

    t=input(1);
    spare_store=input(3);
    length=input(4);
    mission_timepoint=input(5)-length;
    mission_strengh=input(6);


    if (timer+length*mission_strengh<T)&&(t<=mission_timepoint) 
        if spare_store>0
            action=[0 1 0 0 0 0];% take the mission and don't order
        else
            action=[1 0 0 0 0 0];% take the mission and order
        end
        timer=timer+length*mission_strengh;
    else
        if t>mission_timepoint
            if spare_store<1
                action=[0 0  1 0 0 0];% wait and order
            else
                action=[0 0  0 1 0 0];% wait and don't order
            end
        else
            if spare_store==1
                action=[0 0  0 0 1 0];% perform maintenance and order
                timer=0;
            else
                if spare_store==0
                    action=[0 0  1 0 0 0];% wait and order
                else
                    action=[0 0  0 0 0 1];% perform maintenance and don't order
                    timer=0;
                end
            end
        end
        
    end
    

    
    [reward_temp,cost_temp,cost_store,next,spare_queue]=reward_cal(input,action,ini_health,spare_queue,num,states_vector);


end