function [reward_temp,cost_temp,cost_store,next,action,spare_queue]=after_event_action(input,ini_health,spare_queue,num,states_vector)

    t=input(1);
    health=input(2);
    spare_store=input(3);
    length=input(4);
    mission_timepoint=input(5)-length;



    if (health>0)&&(t<=mission_timepoint) 
        if spare_store>0
            action=[0 1 0 0 0 0];% take the mission and don't order
        else
            action=[1 0 0 0 0 0];% take the mission and order
        end
    else
        
        if t>mission_timepoint
            if spare_store<1
                action=[0 0  1 0 0 0];% cannot perform maintenancem, and order
            else
                action=[0 0  0 1 0 0];% cannot perform maintenancem, and donn't order
            end
        else
            if spare_store<=1
                action=[0 0  0 0 1 0];%  perform maintenancem, and order
            else
                action=[0 0  0 0 0 1];% perform maintenance and don't order
            end
        end
    end


    
    [reward_temp,cost_temp,cost_store,next,spare_queue]=reward_cal(input,action,ini_health,spare_queue,num,states_vector);


end