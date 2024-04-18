%% Clear and initialize
clear % 这个语句非常重要，不能删除！！
clc
close all

%% Initial UR5 setting
import py.rtde_receive.RTDEReceiveInterface
import py.rtde_control.RTDEControlInterface
pyrun("import time");

rtde_r = RTDEReceiveInterface("169.254.159.50");
rtde_c = RTDEControlInterface("169.254.159.50");

disp('UR5 Successfully Connected!');
pyrun("time.sleep(1)");

RobotiQ_2F_140_Length = 1e-3 * 232.8;  % m
%% Initial Gripper Setting
% Close COM4 if it's open
if ~isempty(instrfind) 
    fclose(instrfind);
    delete(instrfind);
end

% Define serial port parameters
port = 'COM4'; % Replace 'COM4' with the appropriate COM port on your PC
baudrate = 115200;

% Open serial port
s = serial(port, 'BaudRate', baudrate);
fopen(s);

% Gripper activation request
activation_request = uint8([9, 16, 3, 232, 0, 3, 6, 0, 0, 0, 0, 0, 0, 115, 48]);
fwrite(s, activation_request);
pause(0.01); % Wait for response

%% The RGBD camera communication
host = '192.168.1.1';
port = 5000;
client_socket = tcpclient(host, port);
disp('RGBD Camera Connection Success!');
numElement = 1*4;


%% Set UR5 initial position

initial_TCP = [0.0562 -0.3742 0.4558-RobotiQ_2F_140_Length 0 0 0]; % safe initial TCP position

rtde_c.moveL(initial_TCP, 0.05, 0.05); % Move to initial position
pyrun("time.sleep(4)");

safeBounds_x = initial_TCP(1) + 1e-3*[-30, 30];     %% set the safe bounds for the UR5 robot, prevent human injury!
safeBounds_y = initial_TCP(2) + 1e-3*[-30, 30];
safeBounds_z = initial_TCP(3) + 1e-3*[-50, 50];

safeBounds_rx = initial_TCP(4) + deg2rad([-30, 30]);     %% set the safe bounds for the UR5 robot, prevent human injury!
safeBounds_ry = initial_TCP(5) + deg2rad([-30, 30]);
safeBounds_rz = initial_TCP(6) + deg2rad([-30, 30]);
%% Peform picking up the ring

pickAcc = 0.05;
pickVelo = 0.05;

ringLocation = [-0.1582   -0.3009    0.2779-RobotiQ_2F_140_Length    0    0    0];
rtde_c.moveL(ringLocation, pickAcc, pickVelo);
% pyrun("time.sleep(1)");

% Close gripper
disp('Closing gripper');
close_command = [9, 16, 3, 232, 0, 3, 6, 9, 0, 0, hex2dec('FF'), hex2dec('FF'), hex2dec('FF'), 66, 41];
fwrite(s, close_command);
pause(2); % Wait for gripper to close

ringLocationLift = [-0.1582   -0.3009    0.2779 + 0.03-RobotiQ_2F_140_Length    0    0    0];
rtde_c.moveL(ringLocationLift, pickAcc, pickVelo);
% pyrun("time.sleep(2)");

rtde_c.moveL(initial_TCP, 2*pickAcc, 2*pickVelo);


%% Go to readyPosition

readyLocation = [0.0514   -0.4661    0.4324-RobotiQ_2F_140_Length    0    0    0];
rtde_c.moveL(readyLocation, 2*pickAcc, 2*pickVelo);

pyrun("time.sleep(2)");

%% Insert the ring to finger
% 
% releaselLocation = [0.0514   -0.5533    0.4324-RobotiQ_2F_140_Length    0    0    0];
% rtde_c.moveL(releaselLocation, pickAcc, pickVelo);
% 
% % Open gripper
% disp('Opening gripper');
% open_command = [9, 16, 3, 232, 0, 3, 6, 9, 0, 0, 0, hex2dec('FF'), hex2dec('FF'), 114, 25];
% fwrite(s, open_command);
% pause(2); % Wait for gripper to open
% 
% rtde_c.moveL(initial_TCP, 2*pickAcc, 2*pickVelo);
% pyrun("time.sleep(2)");

%% Execute trajectory
% Move to desired TCP location
safeVelo = 0.1; % Adjust velocity (m/s)
safeAcc = 0.1; % Adjust acceleration (m/s^2)
pauseTime = 0.01; % Reduce pause time between movements (s)
dt = 1.0/500;
lookahead_time = 0.1;
gain = 300;

tic;

while true
    try
        data = read(client_socket, numElement, 'single');  % Receive data
        array = reshape(data, [1,4]);
        disp(array);
        ref_z_current = -1 * array(2);
        ref_x_rotate = 0.5 * array(4);
        % disp('1');
        relativeMotion = [0, 0, ref_z_current];
        relativeRotate = deg2rad([-1 * ref_x_rotate, 0, 0]);
                    % disp('2');
        % Calculate desired TCP position
        desired_TCP = [initial_TCP(1) + relativeMotion(1), ...
                       initial_TCP(2) + relativeMotion(2), ...
                       initial_TCP(3) + relativeMotion(3), ...
                       initial_TCP(4) + relativeRotate(1), ...
                       initial_TCP(5) + relativeRotate(2), ...
                       initial_TCP(6) + relativeRotate(3)];
    
        currentTCP = cellfun(@double, cell(rtde_r.getActualTCPPose()));



        waitTime = toc;

        if waitTime > 5
            disp('Now 5 seconds later');
            % Check if the current TCP is within the safe bounds
            if all(currentTCP(1) >= safeBounds_x(1) && currentTCP(1) <= safeBounds_x(2)) && ...
               all(currentTCP(2) >= safeBounds_y(1) && currentTCP(2) <= safeBounds_y(2)) && ...
               all(currentTCP(3) >= safeBounds_z(1) && currentTCP(3) <= safeBounds_z(2)) && ...
               all(currentTCP(4) >= safeBounds_rx(1) && currentTCP(4) <= safeBounds_rx(2)) && ...
               all(currentTCP(5) >= safeBounds_ry(1) && currentTCP(5) <= safeBounds_ry(2)) && ...
               all(currentTCP(6) >= safeBounds_rz(1) && currentTCP(6) <= safeBounds_rz(2))

                rtde_c.servoL(desired_TCP, safeVelo, safeAcc, dt, lookahead_time, gain);   
            else
                disp('Bound Stop Activited!!!');
            end

            % Pause for a shorter duration between movements
            pause(pauseTime);
        end
    catch
        disp('No Camera Input.')
        continue;
    end
end

%% End control
rtde_c.stopScript();
