disp("MATLAB Simulation Started")

time = (1:20)';
pressure = rand(20,1) * 100;
temperature = rand(20,1) * 50;
fault = repmat("Bearing Fault",20,1);

T = table(time, pressure, temperature, fault);

writetable(T, 'C:/project/backend/matlab/simulation_output.csv');



saveas(gcf, 'C:/project/backend/matlab/graph.png');

disp("CSV Generated")