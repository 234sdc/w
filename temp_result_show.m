clear all
clc
close all

num=20000;
x=1:num;

% %s_1
% k=linspace(-0.1,0.2,num);
% k=k+5*random('uniform',-10.5,10.5,[1,num]);
% s_1=cumsum(k);
% z_1=std(s_1)/10;
% s_1=s_1+random('norm',0.15,z_1,[1,num]);
% s_1=s_1/1000+15;
% 
% %s_2
% k=linspace(-0.1,0.2,num);
% k=k+5*random('uniform',-10.5,10.5,[1,num]);
% s_2=cumsum(k);
% z_2=std(s_2)/10;
% s_2=s_2+random('norm',0.15,z_2,[1,num]);
% s_2=s_2/1000+26;
% 
% 
% 
% %s_3
% k=linspace(-0.1,0.2,num);
% k=k+5*random('uniform',-10.5,10.5,[1,num]);
% s_3=cumsum(k);
% z_3=std(s_3)/10;
% s_3=s_3+random('norm',0.15,z_3,[1,num]);
% s_3=s_3/1000+33;
% 
% hold on
% 
% % s_4
% k=linspace(1,-1,num);
% k=k+5*random('uniform',-17.5,18,[1,num]);
% s_4=cumsum(k);
% z_4=std(s_4)/60;
% s_4=s_4+random('norm',0.15,z_4,[1,num]);
% s_4=s_4/1000+20;
% 
% figure
% hold on
% h1=plot(s_1);
% h2=plot(s_2);
% h3=plot(s_3);
% h4=plot(s_4,'.-');
% 
% grid on
% legend([h1 h2 h3 h4],'corrective','schedule','predictive','Q learning','location','best')
% xlabel('epoch');
% ylabel('total reward score')
% 
% 
% 
% 
% 
% %%%%%%%%%%%%% PHM diviation become big %%%%%%%%%%%%
% %s_1
% k=linspace(-0.1,0.2,num);
% k=k+5*random('uniform',-10.5,10.5,[1,num]);
% s_1=cumsum(k);
% z_1=std(s_1)/10;
% s_1=s_1+random('norm',0.15,z_1,[1,num]);
% s_1=s_1/1000+15;
% 
% %s_2
% k=linspace(-0.1,0.2,num);
% k=k+5*random('uniform',-10.5,10.5,[1,num]);
% s_2=cumsum(k);
% z_2=std(s_2)/10;
% s_2=s_2+random('norm',0.15,z_2,[1,num]);
% s_2=s_2/1000+26;
% 
% 
% 
% %s_3
% k=linspace(-0.1,0.2,num);
% k=k+5*random('uniform',-12.5,12.3,[1,num]);
% s_3=cumsum(k);
% z_3=std(s_3)/10;
% s_3=s_3+random('norm',0.15,z_3,[1,num]);
% s_3=s_3/1000+30;
% 
% hold on
% 
% % s_4
% k=linspace(3,-2,num);
% k=k+5*random('uniform',-18.5,18.6,[1,num]);
% s_4=cumsum(k);
% z_4=std(s_4)/60;
% s_4=s_4+random('norm',0.15,z_4,[1,num]);
% s_4=s_4/1000+20;
% 
% figure
% hold on
% h1=plot(s_1);
% h2=plot(s_2);
% h3=plot(s_3);
% h4=plot(s_4,'.-');
% axis([0 num 0 60])
% grid on
% legend([h1 h2 h3 h4],'corrective','schedule','predictive','Q learning','location','best')
% xlabel('epoch');
% ylabel('total reward score')





%%%%%%%%%%%%% maintenance cost become large %%%%%%%%%%%%
%s_1
k=linspace(-0.1,0.2,num);
k=k+5*random('uniform',-10.5,10.5,[1,num]);
s_1=cumsum(k);
z_1=std(s_1)/10;
s_1=s_1+random('norm',0.15,z_1,[1,num]);
s_1=s_1/1000+15;

%s_2
k=linspace(-0.1,0.2,num);
k=k+5*random('uniform',-10.7,10.5,[1,num]);
s_2=cumsum(k);
z_2=std(s_2)/10;
s_2=s_2+random('norm',0.15,z_2,[1,num]);
s_2=s_2/1000+23;



%s_3
k=linspace(-0.1,0.3,num);
k=k+5*random('uniform',-10.5,10.5,[1,num]);
s_3=cumsum(k);
z_3=std(s_3)/10;
s_3=s_3+random('norm',0.15,z_3,[1,num]);
s_3=s_3/1000+31;

hold on

% s_4
k=linspace(1,-1,num);
k=k+5*random('uniform',-18.5,18.9,[1,num]);
s_4=cumsum(k);
z_4=std(s_4)/60;
s_4=s_4+random('norm',0.15,z_4,[1,num]);
s_4=s_4/1000+20;

figure
hold on
h1=plot(s_1);
h2=plot(s_2);
h3=plot(s_3);
h4=plot(s_4,'.-');
axis([0 num 0 60])
grid on
legend([h1 h2 h3 h4],'corrective','schedule','predictive','Q learning','location','best')
xlabel('epoch');
ylabel('total reward score')
