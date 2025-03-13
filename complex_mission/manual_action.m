function [reward_temp,cost_temp,next,action,spare_queue]=manual_action(input,ini_health,spare_queue,num)
%     [~,action]=max(act);


    t=input(1);
    health=input(2);
    health_sense=health+health/10*randn;
    spare_store=input(3);
    length=input(4);
    mission_timepoint=input(5);
    mission_strengh=input(6);



    resu=health_sense-length*mission_strengh;
    if (resu>0)&&(t<=mission_timepoint) 
        if resu>=(length*mission_strengh)
            action=[0 1 0 0 0 0];% take the mission and don't order
        else
            action=[1 0 0 0 0 0];% take the mission and order
        end
    else
        if resu<=0 % perform maintenance
            if spare_store<1
                action=[0 0  0 0 1 0];% perform maintenance and order
            else
                action=[0 0  0 0 0 1];% perform maintenance and don't order
            end
        else
            if spare_store<1
                action=[0 0  1 0 0 0];% wait and order
            else
                action=[0 0  0 1 0 0];% wait and don't order
            end
        end
    end


    
    [reward_temp,cost_temp,next,spare_queue]=reward_cal(input,action,ini_health,spare_queue,num);


end