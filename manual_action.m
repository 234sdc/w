function [reward_temp,cost_temp,next,action]=manual_action(input,ini_health)
%     [~,action]=max(act);

    t=input(1);
    health=input(2);
    health_sense=health+health/10*randn;
    length=input(3);
    mission_timepoint=input(4);
    mission_strengh=input(5);
    mission_reward=input(6);
    maintenance_time=input(7);
    maintenance_cost=input(8);
    resu=health_sense-length*mission_strengh;
    if (resu>0)&&(t<=mission_timepoint)
        action=[1 0 0];
    else
        if resu<=0
            action=[0 0 1];
        else
            action=[0 1 0];
        end
    end
        [~,act]=max(action);
    switch act
        case 1 % take the mission
            resu=health-length*mission_strengh;
            if (t<=mission_timepoint) % mission is acutally taken
                health=resu;
                if (resu<=0)
                    reward_temp=-mission_reward;
                else
                    reward_temp=mission_reward;
                end
            else
                reward_temp=0;
            end
            t=mission_timepoint;
            cost_temp=0;
        case 2 % wait
            cost_temp=0;
            reward_temp=0;
        case 3 % maintenance
            reward_temp=0;
            cost_temp=maintenance_cost;
            health=ini_health;
            t=t+maintenance_time;
    end

    %% next state
        temp=[t health];
        input(1:2)=[];% remove the health state and t, then reshape input as the 10*6
        input=reshape(input,[6,10]);    
        input=input';   
        z1=mission_profile(1);
        add=[ z1.length  z1.mission_timepoint  z1.mission_strengh  ...
            z1.mission_reward  z1.maintenance_time  z1.maintenance_cost];
        input(1:end-1,:)=input(2:end,:);
        input(end,:)=add;
        input(end,2)=input(end,1)+input(end-1,2);
        next=[temp reshape(input',1,[])];


end