function input_set=mission_profile_all(nums,thre)
% state generation


state_data.length=random('uniform',1,20,[nums,1]);
state_data.mission_timepoint=cumsum(state_data.length);
state_data.mission_strengh=random('uniform',0.8,1.5,[nums,1]);
state_data.mission_reward=random('uniform',1,40,[nums,1]);

for i=1:nums %% randomly generate the very high mission reward with probability thre
    if rand<thre
        state_data.mission_reward(i)=random('uniform',50,100,[1,1]);
    end
end
state_data.corr_m_time=random('uniform',12,18,[nums,1]);
state_data.corr_m_cost=random('uniform',8,12,[nums,1]);
state_data.prev_m_time=random('uniform',4,6,[nums,1]);
state_data.prev_m_cost=random('uniform',2.5,4,[nums,1]);


input_set=[state_data.length  state_data.mission_timepoint  state_data.mission_strengh  ...
state_data.mission_reward state_data.corr_m_time  state_data.corr_m_cost state_data.prev_m_time  state_data.prev_m_cost];
end

