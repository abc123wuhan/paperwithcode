
%%  简单的散点图
%%  swarm c
%图
obs_swarmc1=obs_swarmc_all_nan(ap_swarmc_all_nan<12);obs_swarmc2=obs_swarmc_all_nan(ap_swarmc_all_nan>=12);
%obs_nrlmswarmc1=obs_nrlmswarmc(ap_nrlmswarmc<12);obs_nrlmswarmc2=obs_nrlmswarmc(ap_nrlmswarmc>=12);

pre_swarmc1=pre_swarmc_all_nan(ap_swarmc_all_nan<12);pre_swarmc2=pre_swarmc_all_nan(ap_swarmc_all_nan>=12);
%pre_nrlmswarmc1=pre_nrlmswarmc(ap_nrlmswarmc<12);pre_nrlmswarmc2=pre_nrlmswarmc(ap_nrlmswarmc>=12);
%平静期
%a=isnan(obs_swarmc1);b=find(a>0);pre_swarmc1(b)=[];obs_swarmc1(b)=[];
%plot(obs_nrlmswarmc1,pre_nrlmswarmc1,'.','Color',[0.13 0.55 0.13],'DisplayName','NRLMSISE 2.0');hold on;
plot(obs_swarmc1,pre_swarmc1,'.','Color',[0.898 0 0],'DisplayName','MBiLE');hold on;
plot(obs_swarmc1,obs_swarmc1,'.','Color',[0 0.45 0.74],'DisplayName','Observation');hold on;

coefficients = polyfit(obs_swarmc1, pre_swarmc1, 1);
x_values = linspace(min(obs_swarmc1), max(obs_swarmc1), 100);
y_values = polyval(coefficients, x_values);
%plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_swarmc1- obs_swarmc1).^2)) / sum((obs_swarmc1-mean(obs_swarmc1)).^2));  
text(max(obs_swarmc1), max(pre_swarmc1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.898 0 0],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

% coefficients = polyfit(obs_nrlmswarmc1, pre_nrlmswarmc1, 1);
% x_values = linspace(min(obs_nrlmswarmc1), max(obs_nrlmswarmc1), 100);
% y_values = polyval(coefficients, x_values);
%%plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
% equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
% R2=1 - ((sum((pre_nrlmswarmc1- obs_nrlmswarmc1).^2)) / sum((obs_nrlmswarmc1-mean(obs_nrlmswarmc1)).^2)); 
% text(max(obs_nrlmswarmc1), max(pre_nrlmswarmc1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.13 0.55 0.13],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

xlim([0 0.36])
ylim([0 0.36])
xlabel('Observation [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
ylabel('Prediction [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
%legend({'Observation','MBiLE','NRLMSISE 2.0'},'Location','northwest','NumColumns',1);
legend('Location','northwest','NumColumns',1);
legend('FontName','Times New Roman','FontSize',15,'LineWidth',5);
legend('boxoff');
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
title('Swarm C, quiet period')

%磁暴期
%a=isnan(obs_swarmc2);b=find(a>0);pre_swarmc2(b)=[];obs_swarmc2(b)=[];
%plot(obs_nrlmswarmc2,pre_nrlmswarmc2,'.','Color',[0.13 0.55 0.13],'DisplayName','NRLMSISE 2.0');hold on;
plot(obs_swarmc2,pre_swarmc2,'.','Color',[0.898 0 0],'DisplayName','MBiLE');hold on;
plot(obs_swarmc2,obs_swarmc2,'.','Color',[0 0.45 0.74],'DisplayName','Observation');hold on;

coefficients = polyfit(obs_swarmc2, pre_swarmc2, 1);
x_values = linspace(min(obs_swarmc2), max(obs_swarmc2), 100);
y_values = polyval(coefficients, x_values);
%plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_swarmc2- obs_swarmc2).^2)) / sum((obs_swarmc2-mean(obs_swarmc2)).^2));  %0.9791
text(max(obs_swarmc2), max(pre_swarmc2), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.898 0 0],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

% coefficients = polyfit(obs_nrlmswarmc2, pre_nrlmswarmc2, 1);
% x_values = linspace(min(obs_nrlmswarmc2), max(obs_nrlmswarmc2), 100);
% y_values = polyval(coefficients, x_values);
% %plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
% equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
% R2=1 - ((sum((pre_nrlmswarmc2- obs_nrlmswarmc2).^2)) / sum((obs_nrlmswarmc2-mean(obs_nrlmswarmc2)).^2));  %0.9791
% text(max(obs_nrlmswarmc2), max(pre_nrlmswarmc2), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.13 0.55 0.13],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

xlim([0 0.50])
ylim([0 0.50])
xlabel('Observation [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
ylabel('Prediction [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
%legend({'Observation','MBiLE','NRLMSISE 2.0'},'Location','northwest','NumColumns',1);
legend('Location','northwest','NumColumns',1);
legend('FontName','Times New Roman','FontSize',15,'LineWidth',5);
legend('boxoff');
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
title('Swarm C, storm period')

%图5
obs_swarmc1=obs_swarmc(ap_swarmc<12);
obs_nrlmswarmc1=obs_nrlmswarmc(ap_nrlmswarmc<12);
pre_swarmc1=pre_swarmc(ap_swarmc<12);
pre_nrlmswarmc1=pre_nrlmswarmc(ap_nrlmswarmc<12);

%求obs-pre的标准差
error1_std=std(obs_swarmc1-pre_swarmc1,1);
error2_std=std(obs_nrlmswarmc1-pre_nrlmswarmc1,1);

% 随机选取 length(nrlm)个索引
selected_indices = randperm(length(obs_nrlmswarmc1),length(obs_swarmc1));
% 从 obs 中选取对应的观测值和预测值
selected_obs = obs_nrlmswarmc1(selected_indices);
selected_pre = pre_nrlmswarmc1(selected_indices);
fenzi = sum((obs_swarmc1 - pre_swarmc1).^2);
fenmu = sum((selected_obs - selected_pre).^2);
PE = 1 - (fenzi / fenmu);
%
a1=plot(obs_nrlmswarmc1,pre_nrlmswarmc1,'.','Color',[0.13 0.55 0.13],'DisplayName','NRLMSISE 2.0');hold on;
a2=plot(obs_swarmc1,pre_swarmc1,'.','Color',[0.898 0 0],'DisplayName','MBiLE');hold on;
a3=plot(obs_swarmc1,obs_swarmc1,'.','Color',[0 0.45 0.74],'DisplayName','Observation');hold on;

coefficients = polyfit(obs_swarmc1, pre_swarmc1, 1);
x_values = linspace(min(obs_swarmc1), max(obs_swarmc1), 100);
y_values = polyval(coefficients, x_values);
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_swarmc1- obs_swarmc1).^2)) / sum((obs_swarmc1-mean(obs_swarmc1)).^2));  %0.9791
text(max(obs_swarmc1), max(pre_swarmc1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.898 0 0],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

coefficients = polyfit(obs_nrlmswarmc1, pre_nrlmswarmc1, 1);
x_values = linspace(min(obs_nrlmswarmc1), max(obs_nrlmswarmc1), 100);
y_values = polyval(coefficients, x_values);
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_nrlmswarmc1- obs_nrlmswarmc1).^2)) / sum((obs_nrlmswarmc1-mean(obs_nrlmswarmc1)).^2));  %0.9791
text(max(obs_nrlmswarmc1), max(pre_nrlmswarmc1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.13 0.55 0.13],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

xlim([0 0.4])
xticks([0 0.1 0.2 0.3 0.4]);
ylim([0 0.4])
yticks([0 0.1 0.2 0.3 0.4]);
xlabel('Observation [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
ylabel('Prediction [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
title('Swarm C, ap<12')

h1=legend(a1,'NRLMSISE 2.0');
set(h1,'position',[0.133465 0.829316 0.15177 0.0369],'FontName','Times New Roman', 'Fontsize', 15,'TextColor',[0.13 0.55 0.13]);
legend('boxoff');
ah=axes('position',get(gca,'position'), 'visible','off');
h2=legend(ah,a2,'MBiLE');
set(h2,'position',[0.132575 0.774684 0.15278 0.0369],'FontName','Times New Roman', 'Fontsize', 15,'TextColor',[0.898 0 0]);
legend('boxoff');
ah=axes('position',get(gca,'position'),'visible','off');
h3=legend(ah,a3,'Observation');
set(h3,'position',[0.1329 0.8801 0.1256 0.0369],'FontName','Times New Roman', 'Fontsize', 15,'TextColor',[0 0.45 0.74]);
legend('boxoff');
%% CHAMP
%图7
obs_champ1=obs_champ_all_nan(ap_champ_all_nan<12);obs_champ2=obs_champ_all_nan(ap_champ_all_nan>=12);
%obs_nrlmchamp1=obs_nrlmchamp(ap_nrlmchamp<12);obs_nrlmchamp2=obs_nrlmchamp(ap_nrlmchamp>=12);

pre_champ1=pre_champ_all_nan(ap_champ_all_nan<12);pre_champ2=pre_champ_all_nan(ap_champ_all_nan>=12);
%pre_nrlmchamp1=pre_nrlmchamp(ap_nrlmchamp<12);pre_nrlmchamp2=pre_nrlmchamp(ap_nrlmchamp>=12);
%平静期
%plot(obs_nrlmchamp1,pre_nrlmchamp1,'.','Color',[0.13 0.55 0.13],'DisplayName','NRLMSISE 2.0');hold on;
plot(obs_champ1,pre_champ1,'.','Color',[0.898 0 0],'DisplayName','MBiLE');hold on;
plot(obs_champ1,obs_champ1,'.','Color',[0 0.45 0.74],'DisplayName','Observation');hold on;

coefficients = polyfit(obs_champ1, pre_champ1, 1);
x_values = linspace(min(obs_champ1), max(obs_champ1), 100);
y_values = polyval(coefficients, x_values);
%plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_champ1- obs_champ1).^2)) / sum((obs_champ1-mean(obs_champ1)).^2));  %0.9791
text(max(obs_champ1), max(pre_champ1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.898 0 0],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

% coefficients = polyfit(obs_nrlmchamp1, pre_nrlmchamp1, 1);
% x_values = linspace(min(obs_nrlmchamp1), max(obs_nrlmchamp1), 100);
% y_values = polyval(coefficients, x_values);
%%plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
% equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
% R2=1 - ((sum((pre_nrlmchamp1- obs_nrlmchamp1).^2)) / sum((obs_nrlmchamp1-mean(obs_nrlmchamp1)).^2));  %0.9791
% text(max(obs_nrlmchamp1), max(pre_nrlmchamp1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.13 0.55 0.13],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

xlim([0 4.2])
ylim([-0.01 4.2])
xlabel('Observation [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
ylabel('Prediction [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
%legend({'Observation','MBiLE','NRLMSISE 2.0'},'Location','northwest','NumColumns',1);
legend('Location','northwest','NumColumns',1);
legend('FontName','Times New Roman','FontSize',15,'LineWidth',5);
legend('boxoff');
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
title('CHAMP, quiet period')

%磁暴期
%plot(obs_nrlmchamp2,pre_nrlmchamp2,'.','Color',[0.13 0.55 0.13],'DisplayName','NRLMSISE 2.0');hold on;
plot(obs_champ2,pre_champ2,'.','Color',[0.898 0 0],'DisplayName','MBiLE');hold on;
plot(obs_champ2,obs_champ2,'.','Color',[0 0.45 0.74],'DisplayName','Observation');hold on;

coefficients = polyfit(obs_champ2, pre_champ2, 1);
x_values = linspace(min(obs_champ2), max(obs_champ2), 100);
y_values = polyval(coefficients, x_values);
%plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_champ2- obs_champ2).^2)) / sum((obs_champ2-mean(obs_champ2)).^2));  
text(max(obs_champ2), max(pre_champ2), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.898 0 0],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

% coefficients = polyfit(obs_nrlmchamp2, pre_nrlmchamp2, 1);
% x_values = linspace(min(obs_nrlmchamp2), max(obs_nrlmchamp2), 100);
% y_values = polyval(coefficients, x_values);
% %plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
% equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
% R2=1 - ((sum((pre_nrlmchamp2- obs_nrlmchamp2).^2)) / sum((obs_nrlmchamp2-mean(obs_nrlmchamp2)).^2));  
% text(max(obs_nrlmchamp2), max(pre_nrlmchamp2), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.13 0.55 0.13],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

xlim([0 5.5])
ylim([-0.01 5.5])
xlabel('Observation [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
ylabel('Prediction [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
%legend({'Observation','MBiLE','NRLMSISE 2.0'},'Location','northwest','NumColumns',1);
legend('Location','northwest','NumColumns',1);
legend('FontName','Times New Roman','FontSize',15,'LineWidth',5);
legend('boxoff');
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
title('CHAMP, storm period')

%图5
obs_champ1=obs_champ_all_nan(ap_champ_all_nan<12);
obs_nrlmchamp1=obs_nrlmchamp_all_nan(ap_nrlmchamp_all_nan<12);
pre_champ1=pre_champ_all_nan(ap_champ_all_nan<12);
pre_nrlmchamp1=pre_nrlmchamp_all_nan(ap_nrlmchamp_all_nan<12);

%求obs-pre的标准差
error1_std=std(obs_champ1-pre_champ1,1);
error2_std=nanstd(obs_nrlmchamp1-pre_nrlmchamp1,1)

% 随机选取 length(nrlm)个索引
selected_indices = randperm(length(obs_nrlmchamp1),length(obs_champ1));
% 从 obs 中选取对应的观测值和预测值
selected_obs = obs_nrlmchamp1(selected_indices);
selected_pre = pre_nrlmchamp1(selected_indices);
fenzi = sum((obs_champ1 - pre_champ1).^2);
fenmu = nansum((selected_obs - selected_pre).^2);
PE = 1 - (fenzi / fenmu);
%
a=isnan(obs_nrlmchamp1);b=find(a>0);pre_nrlmchamp1(b)=[];obs_nrlmchamp1(b)=[];
a=find(pre_nrlmchamp1>3.9);pre_nrlmchamp1(a)=[];obs_nrlmchamp1(a)=[];

a1=plot(obs_nrlmchamp1,pre_nrlmchamp1,'.','Color',[0.13 0.55 0.13],'DisplayName','NRLMSISE 2.0');hold on;
a2=plot(obs_champ1,pre_champ1,'.','Color',[0.898 0 0],'DisplayName','MBiLE');hold on;
a3=plot(obs_champ1,obs_champ1,'.','Color',[0 0.45 0.74],'DisplayName','Observation');hold on;

coefficients = polyfit(obs_champ1, pre_champ1, 1);
x_values = linspace(min(obs_champ1), max(obs_champ1), 100);
y_values = polyval(coefficients, x_values);
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_champ1- obs_champ1).^2)) / sum((obs_champ1-mean(obs_champ1)).^2));  
text(max(obs_champ1), max(pre_champ1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.898 0 0],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

coefficients = polyfit(obs_nrlmchamp1, pre_nrlmchamp1, 1);
x_values = linspace(min(obs_nrlmchamp1), max(obs_nrlmchamp1), 100);
y_values = polyval(coefficients, x_values);
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_nrlmchamp1- obs_nrlmchamp1).^2)) / sum((obs_nrlmchamp1-mean(obs_nrlmchamp1)).^2)); 
text(max(obs_nrlmchamp1), max(pre_nrlmchamp1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.13 0.55 0.13],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

xlim([0 4.2])
xticks([0 1 2 3 4]);
ylim([-0.01 4.2])
yticks([0 1 2 3 4]);
xlabel('Observation [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
ylabel('Prediction [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
title('CHAMP, ap<12')

h1=legend(a1,'NRLMSISE 2.0');
set(h1,'position',[0.132815 0.8320125758 0.1513877 0.0369863],'FontName','Times New Roman', 'Fontsize', 15,'TextColor',[0.13 0.55 0.13]);
legend('boxoff');
ah=axes('position',get(gca,'position'), 'visible','off');
h2=legend(ah,a2,'MBiLE');
set(h2,'position',[0.13285 0.78423 0.0950378 0.0369863],'FontName','Times New Roman', 'Fontsize', 15,'TextColor',[0.898 0 0]);
legend('boxoff');
ah=axes('position',get(gca,'position'),'visible','off');
h3=legend(ah,a3,'Observation');
set(h3,'position',[0.1330423 0.88005685 0.12531538 0.0369863],'FontName','Times New Roman', 'Fontsize', 15,'TextColor',[0 0.45 0.74]);
legend('boxoff');
%% GOCE
obs_goce1=obs_goce(ap_goce<12);obs_goce2=obs_goce(ap_goce>=12);
%obs_nrlmgoce1=obs_nrlmgoce(ap_nrlmgoce<12);obs_nrlmgoce2=obs_nrlmgoce(ap_nrlmgoce>=12);

pre_goce1=pre_goce(ap_goce<12);pre_goce2=pre_goce(ap_goce>=12);
%pre_nrlmgoce1=pre_nrlmgoce(ap_nrlmgoce<12);pre_nrlmgoce2=pre_nrlmgoce(ap_nrlmgoce>=12);

%平静期
%plot(obs_nrlmgoce1,pre_nrlmgoce1,'.','Color',[0.13 0.55 0.13],'DisplayName','NRLMSISE 2.0');hold on;
plot(obs_goce1,pre_goce1,'.','Color',[0.898 0 0],'DisplayName','MBiLE');hold on;
plot(obs_goce1,obs_goce1,'.','Color',[0 0.45 0.74],'DisplayName','Observation');hold on;

coefficients = polyfit(obs_goce1, pre_goce1, 1);
x_values = linspace(min(obs_goce1), max(obs_goce1), 100);
y_values = polyval(coefficients, x_values);
%plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_goce1- obs_goce1).^2)) / sum((obs_goce1-mean(obs_goce1)).^2));  
text(max(obs_goce1), max(pre_goce1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.898 0 0],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');


xlim([0 15])
ylim([-0.02 15])
xlabel('Observation [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
ylabel('Prediction [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
%legend({'Observation','MBiLE','NRLMSISE 2.0'},'Location','northwest','NumColumns',1);
legend('Location','northwest','NumColumns',1);
legend('FontName','Times New Roman','FontSize',15,'LineWidth',5);
legend('boxoff');
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
title('GOCE, quiet period')

%磁暴期
%plot(obs_nrlmgoce2,pre_nrlmgoce2,'.','Color',[0.13 0.55 0.13],'DisplayName','NRLMSISE 2.0');hold on;
plot(obs_goce2,pre_goce2,'.','Color',[0.898 0 0],'DisplayName','MBiLE');hold on;
plot(obs_goce2,obs_goce2,'.','Color',[0 0.45 0.74],'DisplayName','Observation');hold on;

coefficients = polyfit(obs_goce2, pre_goce2, 1);
x_values = linspace(min(obs_goce2), max(obs_goce2), 100);
y_values = polyval(coefficients, x_values);
%plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_goce2- obs_goce2).^2)) / sum((obs_goce2-mean(obs_goce2)).^2));  
text(max(obs_goce2), max(pre_goce2), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.898 0 0],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

coefficients = polyfit(obs_nrlmgoce2, pre_nrlmgoce2, 1);
x_values = linspace(min(obs_nrlmgoce2), max(obs_nrlmgoce2), 100);
y_values = polyval(coefficients, x_values);
%plot(x_values, y_values, 'k-', 'LineWidth', 2);%拟合线
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_nrlmgoce2- obs_nrlmgoce2).^2)) / sum((obs_nrlmgoce2-mean(obs_nrlmgoce2)).^2)); 
text(max(obs_nrlmgoce2), max(pre_nrlmgoce2), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.13 0.55 0.13],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

% xlim([0 15])
% ylim([-0.02 15])
xlabel('Observation [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
ylabel('Prediction [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
%legend({'Observation','MBiLE','NRLMSISE 2.0'},'Location','northwest','NumColumns',1);
legend('Location','northwest','NumColumns',1);
legend('FontName','Times New Roman','FontSize',15,'LineWidth',5);
legend('boxoff');
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
title('GOCE, storm period')

%图5
obs_goce1=obs_goce(ap_goce<12);
obs_nrlmgoce1=obs_nrlmgoce(ap_nrlmgoce<12);
pre_goce1=pre_goce(ap_goce<12);
pre_nrlmgoce1=pre_nrlmgoce(ap_nrlmgoce<12);

%求obs-pre的标准差、PE
error1_std=std(obs_goce1-pre_goce1,1)
error2_std=std(obs_nrlmgoce1-pre_nrlmgoce1,1)

% 随机选取 length(nrlm)个索引
selected_indices = randperm(length(obs_goce1), length(obs_nrlmgoce1));
% 从 obs 中选取对应的观测值和预测值
selected_obs = obs_goce1(selected_indices);
selected_pre = pre_goce1(selected_indices);
fenzi = sum((selected_obs - selected_pre).^2);
fenmu = sum((obs_nrlmgoce1 - pre_nrlmgoce1).^2);
PE = 1 - (fenzi / fenmu);
%

a=isnan(obs_nrlmgoce1);b=find(a>0);pre_nrlmgoce1(b)=[];obs_nrlmgoce1(b)=[];
a=find(pre_nrlmgoce1>3.9);pre_nrlmgoce1(a)=[];obs_nrlmgoce1(a)=[];

a1=plot(obs_nrlmgoce1,pre_nrlmgoce1,'.','Color',[0.13 0.55 0.13],'DisplayName','NRLMSISE 2.0');hold on;
a2=plot(obs_goce1,pre_goce1,'.','Color',[0.898 0 0],'DisplayName','MBiLE');hold on;
a3=plot(obs_goce1,obs_goce1,'.','Color',[0 0.45 0.74],'DisplayName','Observation');hold on;

coefficients = polyfit(obs_goce1, pre_goce1, 1);
x_values = linspace(min(obs_goce1), max(obs_goce1), 100);
y_values = polyval(coefficients, x_values);
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_goce1- obs_goce1).^2)) / sum((obs_goce1-mean(obs_goce1)).^2));  
text(max(obs_goce1), max(pre_goce1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.898 0 0],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

coefficients = polyfit(obs_nrlmgoce1, pre_nrlmgoce1, 1);
x_values = linspace(min(obs_nrlmgoce1), max(obs_nrlmgoce1), 100);
y_values = polyval(coefficients, x_values);
equation = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
R2=1 - ((sum((pre_nrlmgoce1- obs_nrlmgoce1).^2)) / sum((obs_nrlmgoce1-mean(obs_nrlmgoce1)).^2));  %0.9791
text(max(obs_nrlmgoce1), max(pre_nrlmgoce1), sprintf('%s\nR^2: %.2f', equation, R2),'Color',[0.13 0.55 0.13],'FontSize',16,'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

ylim([-0.01 15])
yticks([0 5 10 15]);
xlim([0 15])
xticks([0 5 10 15]);
xlabel('Observation [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
ylabel('Prediction [10^{-11} kg/ m^{3}]','FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
title('GOCE, ap<12')

h1=legend(a1,'NRLMSISE 2.0');
set(h1,'position',[0.132815 0.8320125758 0.1513877 0.0369863],'FontName','Times New Roman', 'Fontsize', 15,'TextColor',[0.13 0.55 0.13]);
legend('boxoff');
ah=axes('position',get(gca,'position'), 'visible','off');
h2=legend(ah,a2,'MBiLE');
set(h2,'position',[0.13285 0.78423 0.0950378 0.0369863],'FontName','Times New Roman', 'Fontsize', 15,'TextColor',[0.898 0 0]);
legend('boxoff');
ah=axes('position',get(gca,'position'),'visible','off');
h3=legend(ah,a3,'Observation');
set(h3,'position',[0.1330423 0.88005685 0.12531538 0.0369863],'FontName','Times New Roman', 'Fontsize', 15,'TextColor',[0 0.45 0.74]);
legend('boxoff');
