function state_data=mission_profile(nums,thre)
% state generation


state_data.length=random('uniform',1,10,[nums,1]);
state_data.mission_timepoint=cumsum(state_data.length);
state_data.mission_strengh=random('uniform',0.8,1.5,[nums,1]);
state_data.mission_reward=random('uniform',2,3.5,[nums,1]);

for i=1:nums %% randomly generate the very high mission reward with probability thre
    if rand<thre
        state_data.mission_reward(i)=random('uniform',5,8,[1,1]);
    end
end
state_data.corr_m_time=random('uniform',8,12,[nums,1]);
state_data.corr_m_cost=random('uniform',8,12,[nums,1]);
state_data.prev_m_time=random('uniform',4,6,[nums,1]);
state_data.prev_m_cost=random('uniform',1,3,[nums,1]);


input_set=[state_data.length  state_data.mission_timepoint  state_data.mission_strengh  ...
state_data.mission_reward state_data.corr_m_time  state_data.corr_m_cost state_data.prev_m_time  state_data.prev_m_cost];
state_data=reshape(input_set',1,[]);
end

