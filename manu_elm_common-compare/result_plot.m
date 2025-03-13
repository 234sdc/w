function result_plot(i,action,ind_,result)
    figure;
    hold on
    plot(result(ind_(i-1)+1:ind_(i),5),action(ind_(i-1)+1:ind_(i)),'o')
    h1=plot(result(ind_(i-1)+1:ind_(i),5),result(ind_(i-1)+1:ind_(i),1));
    h2=plot(result(ind_(i-1)+1:ind_(i),5),result(ind_(i-1)+1:ind_(i),2),'.-');
    h3=plot(result(ind_(i-1)+1:ind_(i),5),result(ind_(i-1)+1:ind_(i),3));
    h4=plot(result(ind_(i-1)+1:ind_(i),5),result(ind_(i-1)+1:ind_(i),4));
    legend([h1 h2 h3 h4],'reward','store cost','maintenance cost','mission reward')
    reward_all=sum(result(ind_(i-1)+1:ind_(i),1))
    store_all=sum(result(ind_(i-1)+1:ind_(i),2))
    maintenance_all=sum(result(ind_(i-1)+1:ind_(i),3))
    score=reward_all-store_all-maintenance_all
end