f107_f107a_all = load('f107_f107a_all.txt');  
F107_year_all=f107_f107a_all(:,1);                                             
F107_doy_all=f107_f107a_all(:,2);  
F107_all=f107_f107a_all(:,3);
% F107a_all=f107_f107a_all(:,4);
% P107_all=(F107_all+F107a_all)/2;

%%

maindir_swarmc = "E:\pre_SwarmC_all"; 
%subdir =  dir( [maindir,'\SW*'] );   % 确定子文件夹
subdir_swarmc =  dir( maindir_swarmc );  
subdirpath_swarmc={};
filepath_swarmc={};
data_swarmc={};
data_textdata_swarmc=[];
data_data_swarmc=[];
for i = 1 : length( subdir_swarmc )
    if( isequal( subdir_swarmc( i ).name, '.' ) ||  isequal( subdir_swarmc( i ).name, '..' ) || ~subdir_swarmc( i ).isdir )   
        continue;
    else
    subdirpath_swarmc = fullfile( maindir_swarmc, subdir_swarmc( i ).name, '*.txt' );
    file_swarmc = dir( subdirpath_swarmc );  

    for j = 1 : length( file_swarmc ) 
        filepath_swarmc = fullfile( maindir_swarmc, subdir_swarmc( i ).name, file_swarmc( j ).name);
        data_swarmc= importdata(filepath_swarmc);  
        data_textdata_swarmc=[data_textdata_swarmc;data_swarmc.textdata];
        data_data_swarmc=[data_data_swarmc;data_swarmc.data];
    end
    end
end

ymd_swarmc(obs_swarmc<0)=[];h_swarmc(obs_swarmc<0)=[];hms_swarmc(obs_swarmc<0)=[];pre_swarmc(obs_swarmc<0)=[];error_swarmc(obs_swarmc<0)=[];
glon_swarmc(obs_swarmc<0)=[];glat_swarmc(obs_swarmc<0)=[];doy_swarmc(obs_swarmc<0)=[];ap_swarmc(obs_swarmc<0)=[];obs_swarmc(obs_swarmc<0)=[];

ymd_swarmc(pre_swarmc<0)=[];h_swarmc(pre_swarmc<0)=[];hms_swarmc(pre_swarmc<0)=[];error_swarmc(pre_swarmc<0)=[];obs_swarmc(pre_swarmc<0)=[];
glon_swarmc(pre_swarmc<0)=[];glat_swarmc(pre_swarmc<0)=[];doy_swarmc(pre_swarmc<0)=[];ap_swarmc(pre_swarmc<0)=[];pre_swarmc(pre_swarmc<0)=[];

t1_swarmc=cell2mat(ymd_swarmc);t2_swarmc=cell2mat(hms_swarmc);t=[t1_swarmc,t2_swarmc];
time1_swarmc1=datetime(t,'InputFormat','yyyy-MM-ddHH:mm:ss.SSS');

t1_swarmc_year=str2num(t1_swarmc(:,1:4));
t1_swarmc_month=str2num(t1_swarmc(:,6:7));
t1_swarmc_day=str2num(t1_swarmc(:,9:10));
doy_swarmc=ymd2iday(t1_swarmc_year,t1_swarmc_month,t1_swarmc_day);
error_swarmc=-error_swarmc;
mjd_swarmc=juliandate(time1_swarmc1,'juliandate')-2451545.5;
%计算MLT
r=h_swarmc/1000+6371.2;
[qdglat_swarmc,qdglon_swarmc,apex_lat,MLT_swarmc]=qdipole(mjd_swarmc,r,glat_swarmc,glon_swarmc,'','E:\apexsh_1900-2025.txt');

%识别出断开的地方设置nan值
time=datenum(time1_swarmc_all);%日期时间数组datetime转换为日期数字datenum
i=2;
while i<length(time)
    if(time(i)-time(i-1)>1)
        obs_swarmc_all=[obs_swarmc_all(1:i-1);nan;obs_swarmc_all(i:end)];
        pre_swarmc_all=[pre_swarmc_all(1:i-1);nan;pre_swarmc_all(i:end)];
        error_swarmc_all=[error_swarmc_all(1:i-1);nan;error_swarmc_all(i:end)];
        h_swarmc_all=[h_swarmc_all(1:i-1);nan;h_swarmc_all(i:end)];
        glon_swarmc_all=[glon_swarmc_all(1:i-1);nan;glon_swarmc_all(i:end)];
        glat_swarmc_all=[glat_swarmc_all(1:i-1);nan;glat_swarmc_all(i:end)];
        ap_swarmc_all=[ap_swarmc_all(1:i-1);nan;ap_swarmc_all(i:end)];
        MLT_swarmc_all=[MLT_swarmc_all(1:i-1);nan;MLT_swarmc_all(i:end)];
        doy_swarmc_all=[doy_swarmc_all(1:i-1);nan;doy_swarmc_all(i:end)];
        mjd_swarmc_all=[mjd_swarmc_all(1:i-1);nan;mjd_swarmc_all(i:end)];
        time=[time(1:i-1);(time(i-1)+time(i))/2;time(i:end)];
    end
    i=i+1;
end
obs_swarmc_all_nan=obs_swarmc_all;
pre_swarmc_all_nan=pre_swarmc_all;
error_swarmc_all_nan=error_swarmc_all;
h_swarmc_all_nan=h_swarmc_all;
glon_swarmc_all_nan=glon_swarmc_all;
glat_swarmc_all_nan=glat_swarmc_all;
ap_swarmc_all_nan=ap_swarmc_all;
MLT_swarmc_all_nan=MLT_swarmc_all;
doy_swarmc_all_nan=doy_swarmc_all;
mjd_swarmc_all_nan=mjd_swarmc_all;

time1=datestr(time,'yyyy-mm-dd HH:MM:SS.FFF');
t1=time1(:,1:10);t2=time1(:,12:end);t3=[t1,t2];
time1_swarmc_all_nan=datetime(t3,'InputFormat','yyyy-MM-ddHH:mm:ss.SSS');

%%   P107  swarmc 2014.2.1-2020.9.30     32 274          
% P107_swarmc=P107_all(5146:7579);
% P107_year_swarmc=P107_year_all(5146:7579);
% P107_doy_swarmc=P107_doy_all(5146:7579);

F107_swarmc=F107_all(5146:7579);
F107_year_swarmc=F107_year_all(5146:7579);
F107_doy_swarmc=F107_doy_all(5146:7579);

a=datenum(time1_swarmc_all_nan);
b=datestr(a,'yyyy-mm-dd HH:MM:SS.FFF');
c1=str2num(b(:,1:4));c2=str2num(b(:,6:7));c3=str2num(b(:,9:10));
doy=ymd2iday(c1,c2,c3);

% swarmc_P107=zeros(length(time1_swarmc_all_nan),1);
swarmc_f107=zeros(length(time1_swarmc_all_nan),1);
for i = 1:length(F107_swarmc)
    ia=find(c1==F107_year_swarmc(i) & doy==F107_doy_swarmc(i)); 
    swarmc_f107(ia)=F107_swarmc(i)*ones(length(ia),1); 
end

subplot(400,1,[1:100]);
% plot(time1_swarmc_all_nan,swarmc_f107,'black');hold on;
plot(time1_swarmc_all_nan,swarmc_f107,'black');hold on;
ylabel({'F10.7';'[sfu]'},'FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
set(gca,'xtick',[]);
yticks([60 110 160 210]);
yticklabels({'60','110','160','210'});
ylim([55 230]);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2014-01-01 00:00:00'),datetime('2020-12-01 00:00:00')]);%指定x范围
title('Swarm C');
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

%高度
subplot(400,1,[101:200]);
plot(time1_swarmc_all_nan,h_swarmc_all_nan/1000,'Color',[0.75 0.75 0.75]);hold on;%0.41 0.41 0.41
ylabel({'Altitude';'[km]'},'FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
set(gca,'xtick',[]);
yticks([440 480 520]);
yticklabels({'440','480','520'});
ylim([430 530]);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2014-01-01 00:00:00'),datetime('2020-12-01 00:00:00')]);%指定x范围
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

%pre obs
% m=obs_queshi;
% n=pre_queshi;
% n(183232)=m(183232);n(159461)=m(159461);n(109138)=m(109138);n(117899)=m(117899);n(58856)=m(58856);n(16741)=m(16741);
% n(73246)=m(73246);n(80845)=m(80845);n(85347)=m(85347);

subplot(400,1,[201:300]);

plot(time1_swarmc_all_nan,obs_swarmc_all_nan,'Color',[0 0.45 0.74],'DisplayName','Observation');hold on;  %
plot(time1_swarmc_all_nan,pre_swarmc_all_nan,'Color',[0.898 0 0],'DisplayName','Prediction');hold on;  %0 0.45 0.74

ylabel({'Atmosphere density';'[10^{-11} kg/ m^{3}]'},'FontName','Times New Roman','FontSize',18);
legend({'Observation','Prediction'},'Location','northwest','NumColumns',1);
legend('FontName','Times New Roman','FontSize',15,'LineWidth',5);
legend('boxoff');
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
set(gca,'xtick',[]);
yticks([0 0.2 0.4 0.6]);
yticklabels({'0','0.2','0.4','0.6'});
ylim([-0.02 0.62]);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2014-01-01 00:00:00'),datetime('2020-12-01 00:00:00')]);%指定x范围
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

%误差
subplot(400,1,[301:400]);
plot(time1_swarmc_all_nan,error_swarmc_all_nan,'Color',[0.984 0.52 0]);hold on;
xlabel('Year','FontName','Times New Roman','FontSize',18);
ylabel({'Δ\rho';'[10^{-11} kg/ m^{3}]'},'FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
yticks([-0.4 -0.2 0 0.2]);
yticklabels({'-0.4','-0.2','0','0.2',});
ylim([-0.5 0.25]);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2014-01-01 00:00:00'),datetime('2020-12-01 00:00:00')]);%指定x范围
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

annotation('textbox',...
[0.030555555555556 0.885620377978633 0.09 0.038373048479868],...
'String','(a)',...
'HorizontalAlignment','center',...
'FontSize',18,...
'FontName','Times New Roman',...
'FitBoxToText','off',...
'EdgeColor','none'); %标注(a)

%%  champ数据的完整版 即champ_all 在obs_nrlm20_error中生成保存的，这里画图的时候是直接用的
%CHAMP
maind_champir_champ = "E:\pre_champ_all"; %"E:\pre_champ_all"
%subdir =  dir( [maind_champir,'\SW*'] );   % 确定子文件夹
subdir_champ =  dir( maind_champir_champ );   % 确定子文件夹
subdirpath_champ={};
filepath_champ={};
data_champ={};
data_textdata_champ=[];
data_data_champ=[];
for i = 1 : length( subdir_champ )
    if( isequal( subdir_champ( i ).name, '.' ) ||  isequal( subdir_champ( i ).name, '..' ) || ~subdir_champ( i ).isdir )   % 如果不是目录就跳过
        continue;
    else
    subdirpath_champ = fullfile( maind_champir_champ, subdir_champ( i ).name, '*.txt' );
    file_champ = dir( subdirpath_champ );   % 在这个子文件夹下找后缀为txt的文件
% 对目标文件进行读取
    for j = 1 : length( file_champ ) 
        filepath_champ = fullfile( maind_champir_champ, subdir_champ( i ).name, file_champ( j ).name); % filepath只保存最后一个子文件夹的路径
        data_champ= importdata(filepath_champ);  
        data_textdata_champ=[data_textdata_champ;data_champ.textdata];
        data_data_champ=[data_data_champ;data_champ.data];
    end
    end
end
obs_champ=data_data_champ(:,3);pre_champ=data_data_champ(:,2);error_champ=data_data_champ(:,4);
h_champ=data_data_champ(:,7);
ymd_champ=data_textdata_champ(:,3);
hms_champ=data_textdata_champ(:,4);

glon_champ=data_data_champ(:,5);  %glon
glat_champ=data_data_champ(:,6); %glat
doy_champ=data_data_champ(:,1);  %doy
ap_champ=data_data_champ(:,8);  %ap
kp_champ=data_data_champ(:,9);  %kp

%去除预测异常值
ymd_champ(obs_champ<0)=[];h_champ(obs_champ<0)=[];hms_champ(obs_champ<0)=[];pre_champ(obs_champ<0)=[];error_champ(obs_champ<0)=[];
glon_champ(obs_champ<0)=[];glat_champ(obs_champ<0)=[];doy_champ(obs_champ<0)=[];ap_champ(obs_champ<0)=[];kp_champ(obs_champ<0)=[];obs_champ(obs_champ<0)=[];

ymd_champ(pre_champ<0)=[];h_champ(pre_champ<0)=[];hms_champ(pre_champ<0)=[];error_champ(pre_champ<0)=[];obs_champ(pre_champ<0)=[];
glon_champ(pre_champ<0)=[];glat_champ(pre_champ<0)=[];doy_champ(pre_champ<0)=[];ap_champ(pre_champ<0)=[];kp_champ(pre_champ<0)=[];pre_champ(pre_champ<0)=[];

error_champ=-error_champ;

t1_champ=cell2mat(ymd_champ);t2_champ=cell2mat(hms_champ);t=[t1_champ,t2_champ];
time1_champ=datetime(t,'InputFormat','yyyy-MM-ddHH:mm:ss.SSS');

t1_champ_year=str2num(t1_champ(:,1:4));
t1_champ_month=str2num(t1_champ(:,6:7));
t1_champ_day=str2num(t1_champ(:,9:10));
doy_champ=ymd2iday(t1_champ_year,t1_champ_month,t1_champ_day);

mjd_champ=juliandate(time1_champ,'juliandate')-2451545.5;
%计算MLT
r=h_champ/1000+6371.2;
[qdlat_champ,qdlon_champ,apex_lat,MLT_champ]=qdipole(mjd_champ,r,glat_champ,glon_champ,'','E:\apexsh_1900-2025.txt');

%%  识别出断开的地方设置nan值
time=datenum(time1_champ_all);%日期时间数组datetime转换为日期数字datenum
i=2;
while i<length(time)
    if(time(i)-time(i-1)>1)
        obs_champ_all_nan=[obs_champ_all(1:i-1);nan;obs_champ_all(i:end)];
        pre_champ_all_nan=[pre_champ_all(1:i-1);nan;pre_champ_all(i:end)];
        error_champ_all_nan=[error_champ_all(1:i-1);nan;error_champ_all(i:end)];
        glat_champ_all_nan=[glat_champ_all(1:i-1);nan;glat_champ_all(i:end)];
        glon_champ_all_nan=[glon_champ_all(1:i-1);nan;glon_champ_all(i:end)];
        MLT_champ_all_nan=[MLT_champ_all(1:i-1);nan;MLT_champ_all(i:end)];
        ap_champ_all_nan=[ap_champ_all(1:i-1);nan;ap_champ_all(i:end)];
        h_champ_all_nan=[h_champ_all(1:i-1);nan;h_champ_all(i:end)];
  
        time=[time(1:i-1);(time(i-1)+time(i))/2;time(i:end)];
    end
    i=i+1;
end
time1=datestr(time,'yyyy-mm-dd HH:MM:SS.FFF');
t1=time1(:,1:10);t2=time1(:,12:end);t3=[t1,t2];
time1_champ_all_nan=datetime(t3,'InputFormat','yyyy-MM-ddHH:mm:ss.SSS');
%%   P107  champ 2014.2.1-2020.9.30     32 274          
F107_champ=F107_all(490:3900);
F107_year_champ=F107_year_all(490:3900);
F107_doy_champ=F107_doy_all(490:3900);

a=datenum(time1_champ_all_nan);
b=datestr(a,'yyyy-mm-dd HH:MM:SS.FFF');
c1=str2num(b(:,1:4));c2=str2num(b(:,6:7));c3=str2num(b(:,9:10));
doy=ymd2iday(c1,c2,c3);

champ_F107=zeros(length(time1_champ_all_nan),1);
for i = 1:length(F107_champ)
    ia=find(c1==F107_year_champ(i) & doy==F107_doy_champ(i)); 
    
    champ_F107(ia)=F107_champ(i)*ones(length(ia),1); 
end
%P107图  champ 2001.5.4-2010.9.4    124  247

subplot(400,1,[1:100]);
plot(time1_champ_all_nan,champ_F107,'black');hold on;
ylabel({'F10.7';'[sfu]'},'FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
set(gca,'xtick',[]);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2001-02-01 00:00:00'),datetime('2010-11-01 00:00:00')]);%指定x范围 
yticks([100 200 300]);%指定x范围
yticklabels({'100','200','300'});
ylim([50 300]);
title('CHAMP');
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

%高度
subplot(400,1,[101:200]);
%plot(time1_champ,h_champ/1000,'Color',[0.75 0.75 0.75]);hold on;
plot(time1_champ_all_nan,h_champ_all_nan/1000,'Color',[0.75 0.75 0.75]);hold on;
ylabel({'Altitude';'[km]'},'FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
set(gca,'xtick',[]);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2001-02-01 00:00:00'),datetime('2010-11-01 00:00:00')]);%指定x范围 
yticks([220 300 400 500]);
yticklabels({'220','300','400','500'});
ylim([210 510]);
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

%pre obs
subplot(400,1,[201:300]);
% plot(time1_champ,obs_champ,'Color',[0 0.45 0.74],'DisplayName','Observation');hold on;  %
% plot(time1_champ,pre_champ,'Color',[0.898 0 0],'DisplayName','Prediction');hold on;  %0 0.45 0.74
plot(time1_champ_all_nan,obs_champ_all_nan,'Color',[0 0.45 0.74],'DisplayName','Observation');hold on;  %
plot(time1_champ_all_nan,pre_champ_all_nan,'Color',[0.898 0 0],'DisplayName','Prediction');hold on;  %0 0.45 0.74
ylabel({'Atmosphere density';'[10^{-11} kg/ m^{3}]'},'FontName','Times New Roman','FontSize',18);
legend({'Observation','Prediction'},'Location','northwest','NumColumns',1);
legend('FontName','Times New Roman','FontSize',15,'LineWidth',5);
legend('boxoff');
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2001-02-01 00:00:00'),datetime('2010-11-01 00:00:00')]);%指定x范围 
yticks([0 2 4]);
yticklabels({'0','2','4'});
ylim([-0.2 5.5]);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
set(gca,'xtick',[]);
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

%误差
subplot(400,1,[301:400]);
% plot(time1_champ,error_champ,'Color',[0.984 0.52 0]);hold on;
plot(time1_champ_all_nan,error_champ_all_nan,'Color',[0.984 0.52 0]);hold on;
xlabel('Year','FontName','Times New Roman','FontSize',18);
ylabel({'Δ\rho';'[10^{-11} kg/ m^{3}]'},'FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2001-02-01 00:00:00'),datetime('2010-11-01 00:00:00')]);%指定x范围 
yticks([ -2 -1 0 1 2]);
yticklabels({'-2','-1','0','1','2'});
ylim([-2.9 2.4]);
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

annotation('textbox',...
[0.030555555555556 0.885620377978633 0.09 0.038373048479868],...
'String','(b)',...
'HorizontalAlignment','center',...
'FontSize',18,...
'FontName','Times New Roman',...
'FitBoxToText','off',...
'EdgeColor','none'); 

%%
%GOCE
maind_champir_goce = "E:\pre_goce_all"; 
%subdir =  dir( [maind_champir,'\SW*'] );   % 确定子文件夹
subdir_goce =  dir( maind_champir_goce );   % 确定子文件夹
subdirpath_goce={};
filepath_goce={};
data_goce={};
data_textdata_goce=[];
data_data_goce=[];
for i = 1 : length( subdir_goce )
    if( isequal( subdir_goce( i ).name, '.' ) ||  isequal( subdir_goce( i ).name, '..' ) || ~subdir_goce( i ).isdir )   % 如果不是目录就跳过
        continue;
    else
    subdirpath_goce = fullfile( maind_champir_goce, subdir_goce( i ).name, '*.txt' );
    file = dir( subdirpath_goce );   % 在这个子文件夹下找后缀为txt的文件
% 对目标文件进行读取
    for j = 1 : length( file ) 
        filepath_goce = fullfile( maind_champir_goce, subdir_goce( i ).name, file( j ).name); % filepath只保存最后一个子文件夹的路径
        data_goce= importdata(filepath_goce);  
        data_textdata_goce=[data_textdata_goce;data_goce.textdata];
        data_data_goce=[data_data_goce;data_goce.data];
    end
    end
end

%去除预测异常值
ind_goce=find(data_data_goce(:,4)>3.8 | data_data_goce(:,4)<-1.8);
obs_goce=data_data_goce(:,3);obs_goce(ind_goce)=[];
pre_goce=data_data_goce(:,2);pre_goce(ind_goce)=[];
error_goce=data_data_goce(:,4);error_goce(ind_goce)=[];
h_goce=data_data_goce(:,7);h_goce(ind_goce)=[];
ymd_goce=data_textdata_goce(:,3);ymd_goce(ind_goce)=[];
hms_goce=data_textdata_goce(:,4);hms_goce(ind_goce)=[];
ap_goce=data_data_goce(:,8);ap_goce(ind_goce)=[];

glon_goce=data_data_goce(:,5);glon_goce(ind_goce)=[];  %glon
glat_goce=data_data_goce(:,6);glat_goce(ind_goce)=[];  %glat
doy_goce=data_data_goce(:,1);doy_goce(ind_goce)=[];  %doy

ymd_goce(obs_goce<0)=[];h_goce(obs_goce<0)=[];hms_goce(obs_goce<0)=[];pre_goce(obs_goce<0)=[];error_goce(obs_goce<0)=[];
glon_goce(obs_goce<0)=[];glat_goce(obs_goce<0)=[];doy_goce(obs_goce<0)=[];ap_goce(obs_goce<0)=[];obs_goce(obs_goce<0)=[];

ymd_goce(pre_goce<0)=[];h_goce(pre_goce<0)=[];hms_goce(pre_goce<0)=[];error_goce(pre_goce<0)=[];obs_goce(pre_goce<0)=[];
glon_goce(pre_goce<0)=[];glat_goce(pre_goce<0)=[];doy_goce(pre_goce<0)=[];ap_goce(pre_goce<0)=[];pre_goce(pre_goce<0)=[];

error_goce=-error_goce;

t1_goce=cell2mat(ymd_goce);t2_goce=cell2mat(hms_goce);t=[t1_goce,t2_goce];
time1_goce=datetime(t,'InputFormat','yyyy-MM-ddHH:mm:ss.SSS');

t1_goce_year=str2num(t1_goce(:,1:4));
t1_goce_month=str2num(t1_goce(:,6:7));
t1_goce_day=str2num(t1_goce(:,9:10));
doy_goce=ymd2iday(t1_goce_year,t1_goce_month,t1_goce_day); 

mjd_goce=juliandate(time1_goce,'juliandate')-2451545.5;
%计算MLT
r=h_goce/1000+6371.2;
glon_goce=glon_goce-180;
[lat_now,lon_now,apex_lat,MLT_goce]=qdipole(mjd_goce,r,glat_goce,glon_goce,'','E:\apexsh_1900-2025.txt');

%%  识别出断开的地方设置nan值
time=datenum(time1_goce);%日期时间数组datetime转换为日期数字datenum
i=2;
while i<length(time)
    if(time(i)-time(i-1)>1)
        obs_goce=[obs_goce(1:i-1);nan;obs_goce(i:end)];
        pre_goce=[pre_goce(1:i-1);nan;pre_goce(i:end)];
        error_goce=[error_goce(1:i-1);nan;error_goce(i:end)];
        glat_goce=[glat_goce(1:i-1);nan;glat_goce(i:end)];
        glon_goce=[glon_goce(1:i-1);nan;glon_goce(i:end)];
        MLT_goce=[MLT_goce(1:i-1);nan;MLT_goce(i:end)];
        doy_goce=[doy_goce(1:i-1);nan;doy_goce(i:end)];
        ap_goce=[ap_goce(1:i-1);nan;ap_goce(i:end)];
        h_goce=[h_goce(1:i-1);nan;h_goce(i:end)];
        lat_now=[lat_now(1:i-1);nan;lat_now(i:end)];
        lon_now=[lon_now(1:i-1);nan;lon_now(i:end)];
        mjd_goce=[mjd_goce(1:i-1);nan;mjd_goce(i:end)];
        time=[time(1:i-1);(time(i-1)+time(i))/2;time(i:end)];
    end
    i=i+1;
end
time1=datestr(time,'yyyy-mm-dd HH:MM:SS.FFF');
t1=time1(:,1:10);t2=time1(:,12:end);t3=[t1,t2];
time1_goce=datetime(t3,'InputFormat','yyyy-MM-ddHH:mm:ss.SSS');

%%   P107  goce  2009.11.1-2013.10.20   305  293
F107_goce=F107_all(3593:5042);
F107_year_goce=F107_year_all(3593:5042);
F107_doy_goce=F107_doy_all(3593:5042);

a=datenum(time1_goce);
b=datestr(a,'yyyy-mm-dd HH:MM:SS.FFF');
c1=str2num(b(:,1:4));c2=str2num(b(:,6:7));c3=str2num(b(:,9:10));
doy=ymd2iday(c1,c2,c3);

goce_F107=zeros(length(time1_goce),1);
for i = 1:length(F107_goce)
    ia=find(c1==F107_year_goce(i) & doy==F107_doy_goce(i)); 
    
    goce_F107(ia)=F107_goce(i)*ones(length(ia),1); 
end

%P107图  goce
subplot(400,1,[1:100]);
plot(time1_goce,goce_F107,'black');hold on;%time1_goce,
ylabel({'F10.7';'[sfu]'},'FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
set(gca,'xtick',[]);
yticks([70 110 150 190]);
yticklabels({'70','110','150','190'});
ylim([60 200]);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2009-9-01 00:00:00'),datetime('2013-12-01 00:00:00')]);%指定x范围
title('GOCE');
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

%高度
subplot(400,1,[101:200]);
plot(time1_goce,h_goce/1000,'Color',[0.75 0.75 0.75]);hold on;
ylabel({'Altitude';'[km]'},'FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
set(gca,'xtick',[]);
yticks([220 240 260 280 300]);
yticklabels({'220','240','260','280','300'});
ylim([215 305]);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2009-9-01 00:00:00'),datetime('2013-12-01 00:00:00')]);%指定x范围
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

%pre obs
subplot(400,1,[201:300]);
% plot(time1_goce,obs_goce);hold on;  %
% plot(time1_goce,pre_goce); %0 0.45 0.74
plot(time1_goce,obs_goce,'Color',[0 0.45 0.74],'DisplayName','Observation');hold on;  %
plot(time1_goce,pre_goce,'Color',[0.898 0 0],'DisplayName','Prediction');hold on;  %0 0.45 0.74
ylabel({'Atmosphere density';'[10^{-11} kg/ m^{3}]'},'FontName','Times New Roman','FontSize',18);
legend({'Observation','Prediction'},'Location','northwest','NumColumns',1);
legend('FontName','Times New Roman','FontSize',15,'LineWidth',5);
legend('boxoff');
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
set(gca,'xtick',[]);
yticks([0 5 10 15]);
yticklabels({'0','5','10','15'});
ylim([-0.5 19]);
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2009-9-01 00:00:00'),datetime('2013-12-01 00:00:00')]);%指定x范围
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

%误差
%error_goce=-error_goce;  %预测-观测
subplot(400,1,[301:400]);
plot(time1_goce,error_goce,'Color',[0.984 0.52 0]);hold on;
xlabel('Year','FontName','Times New Roman','FontSize',18);
ylabel({'Δ\rho';'[10^{-11} kg/ m^{3}]'},'FontName','Times New Roman','FontSize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',0.5);
yticks([ -8 -4 0 4 8]);
yticklabels({'-8','-4','0','4','8'});
ylim([-9.2 9.2]);%-2 4.4
xtickformat('yyyy'); %指定 x 轴标签格式
xlim([datetime('2009-9-01 00:00:00'),datetime('2013-12-01 00:00:00')]);%指定x范围
box off
ax2 = axes('Position',get(gca,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
hold off;

annotation('textbox',...
[0.030555555555556 0.885620377978633 0.09 0.038373048479868],...
'String','(c)',...
'HorizontalAlignment','center',...
'FontSize',18,...
'FontName','Times New Roman',...
'FitBoxToText','off',...
'EdgeColor','none'); 

for i=1
    plot([24*i 24*i],[0 8],'-.','color',[.5 .5 .5]);
end
