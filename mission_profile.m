function state_data=mission_profile(nums)
% state generation


state_data.length=random('uniform',1,10,[nums,1]);
state_data.mission_timepoint=cumsum(state_data.length);
state_data.mission_strengh=random('uniform',0.8,1.5,[nums,1]);
state_data.mission_reward=random('uniform',1,15,[nums,1]);
state_data.maintenance_time=random('uniform',1,20,[nums,1]);
state_data.maintenance_cost=random('uniform',1,10,[nums,1]);


end