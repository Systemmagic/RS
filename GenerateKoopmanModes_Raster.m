%% Generate Koopman PM2.5 Modes 
% Modified from GenerateKoopmanModes_811.m to handle GeoTIFF data
% Minimal modification applied to adapt from Station Points to Grid Pixels

function [Psi] = GenerateKoopmanModes_Raster(dataFolder, mode1, mode2)
%% Load Data
clc; close all;
disp('Loading Raster Data Set...');
tic

% --- 修改 1: 数据加载逻辑适配栅格 ---
% 理由: 原代码读取 Excel 站点数据，现需读取 Python/GEE 导出的 28 张 GeoTIFF
fileList = dir(fullfile('CZT_PM25_Daily/', '*.tif'));
% 按文件名排序，确保时间顺序 (01, 02, ... 28)
[~, idx] = sort({fileList.name});
fileList = fileList(idx);

if isempty(fileList)
    error('未找到TIFF文件，请检查路径');
end

% 读取第一张图以获取元数据 (尺寸、地理信息、掩膜)
info = geotiffinfo(fullfile(fileList(1).folder, fileList(1).name));
[base_img, R] = geotiffread(fullfile(fileList(1).folder, fileList(1).name));
[H, W] = size(base_img);

% 构建掩膜 (Masking)
% 理由: 对应 Python 中的 valid_idx，剔除边界外的 NaN 值，只保留有效区域计算 DMD，减少内存消耗
mask = ~isnan(base_img); 
valid_idx = find(mask); % 获取有效像素的一维索引
nbx = length(valid_idx); % 空间维度 N (像素数)
nbt = length(fileList);  % 时间维度 T (天数)

% 构建数据矩阵 Data
Data = zeros(nbx, nbt);
for t = 1:nbt
    img = geotiffread(fullfile(fileList(t).folder, fileList(t).name));
    Data(:, t) = img(valid_idx); % 拉直并提取有效像素
end

% 参数设置 (沿用原代码逻辑)
delay = 3;     % 理由: Python版使用了 Hankel 延迟来增强动力学捕捉，这里设为3天
delt = 1;      % 时间步长: 1天
dtype = 'Mean';
hwy = 'RasterMap'; 

toc

%% Compute KMD and Sort Modes (基本保持不变)
disp('Computing KMD via Hankel-DMD...');
tic
% 去除时间平均值 (中心化)
Avg = mean(Data, 2);
Data_Centered = Data - repmat(Avg, 1, nbt);

% 调用核心 DMD 函数
[eigval, Modes1, bo] = H_DMD(Data_Centered, delay); 
toc

disp('Sorting Modes...');
tic
% 计算频率 omega
omega = log(diag(eigval)) ./ delt; 

% 按幅度排序 (Amplitude Sorting)
% 理由: 相比原代码按频率排序，按幅度排序更能找出主导模态(Mode 1, 2)

magnitudes = abs(bo);
[~, Im] = sort(magnitudes, 'descend'); 

omega = omega(Im); 
Modes1 = Modes1(:, Im); 
bo = bo(Im);
eigval_sorted = diag(eigval); 
eigval_sorted = eigval_sorted(Im);
toc

%% Compute and Plot Modes 
disp('Computing and Plotting Modes...');
tic
time = (0:nbt-1) * delt;

% 准备绘图窗口
figure('units','normalized','outerposition',[0 0 1 1]);

for i = mode1:mode2
    % 计算模态的时间演化 (Time Dynamics)
    omeganow = omega(i);
    bnow = bo(i);
    psi_time = zeros(1, nbt);
    for t = 1:length(time)
        psi_time(:, t) = exp(omeganow * time(t)) * bnow;
    end
    
    % 重构时空矩阵 (Space-Time Reconstruction)
    % Modes1(1:nbx, i) 是空间向量， psi_time 是时间向量
    Psi_vec = Modes1(1:nbx, i) * psi_time; 
    
    % 取实部 (Real Part)，对应 Python 的 np.real()
    % 这里我们只取第一天 (t=1) 的空间形态进行展示，或者取时间平均
    % 为了展示模态的空间结构，我们通常看其对应的空间向量 Modes1
    spatial_mode_vec = real(Modes1(1:nbx, i));
    
    % --- 修改 2: 空间重构与平滑 (Reshape & Smooth) ---
    % 理由: 1. 将 1D 向量还原回 2D 地图
    %       2. 使用 imgaussfilt 消除 ERA5 数据带来的"方块效应" (对应 Python 的 ndimage.gaussian_filter)
    
    mode_map = nan(H, W);
    mode_map(valid_idx) = spatial_mode_vec; % 填回有效值
    
    % 高斯平滑处理 (Sigma=1.5 约等于 Python 中的 sigma=1.5)
    % 注意: 为了不平滑 NaN 边界，先填 0，平滑后再设回 NaN
    temp_map = mode_map;
    temp_map(isnan(temp_map)) = 0;
    smoothed_map = imgaussfilt(temp_map, 1.5);
    smoothed_map(isnan(mode_map)) = nan; % 恢复掩膜
    
    % --- 修改 3: 可视化绘制 (Imagesc instead of Surfc) ---
    % 理由: 栅格数据适合用 imagesc 俯视热力图，而不是 3D surfc
    
    subplot(1, 2, 1); % 左图：单位圆
    theta = linspace(0, 2*pi, 100);
    plot(cos(theta), sin(theta), 'k--'); hold on;
    scatter(real(eigval_sorted), imag(eigval_sorted), 'bo');
    plot(real(eigval_sorted(i)), imag(eigval_sorted(i)), 'r*', 'MarkerSize', 15, 'LineWidth', 2); % 标出当前模态
    axis equal; grid on;
    title(['Eigenvalues (Red: Mode #' num2str(i) ')']);
    xlabel('Real(\lambda)'); ylabel('Imag(\lambda)');
    hold off;

    subplot(1, 2, 2); % 右图：空间模态
    % 使用 mapshow 或 imagesc
    % 为了简单起见，使用 imagesc 并翻转 Y 轴以匹配地图方向
    imagesc(smoothed_map); 
    set(gca, 'YDir', 'normal'); 
    colormap(jet); % 或者用 redblue 自定义色带
    colorbar;
    caxis([-max(abs(smoothed_map(:))), max(abs(smoothed_map(:)))]); % 对称色标
    
    % 标题包含物理意义 (频率/周期)
    freq = imag(omeganow) / (2*pi);
    period = 1 / abs(freq);
    if period > 1000, period_str = 'Inf'; else, period_str = num2str(period, '%.1f'); end
    decay = real(omeganow);
    
    title({['Mode #' num2str(i) ' (Smoothed)'], ...
           ['\lambda = ' num2str(eigval_sorted(i), '%.3f') ...
            ' | T \approx ' period_str ' days | Decay: ' num2str(decay, '%.3f')]}, ...
            'Interpreter', 'tex', 'FontSize', 14);
    axis off; axis equal;
    
    pause(5); % 暂停以便观察
end

disp('All Done');
end