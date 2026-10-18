%% Timetables with TimeStep (row times are saved as origin and step)

data1 = [1;2;3];

timetable_step_millis = timetable(data1, 'TimeStep', milliseconds(10), 'StartTime', datetime(2020,1,1,0,0,0.25));
timetable_step_start_offset = timetable(data1, 'TimeStep', seconds(1), 'StartTime', datetime(2020,1,1,0,0,0.25));
timetable_step_years = timetable(data1, 'TimeStep', years(1), 'StartTime', datetime(2020,1,1));
timetable_step_caldays = timetable(data1, 'TimeStep', caldays(1), 'StartTime', datetime(2020,1,1,12,30,0));
timetable_step_calmonths = timetable(data1, 'TimeStep', calmonths(1), 'StartTime', datetime(2020,1,15,12,30,0));

%% Save

data.timetable_step_millis = timetable_step_millis;
data.timetable_step_start_offset = timetable_step_start_offset;
data.timetable_step_years = timetable_step_years;
data.timetable_step_caldays = timetable_step_caldays;
data.timetable_step_calmonths = timetable_step_calmonths;

save('test_timetable_steps_v7.mat', '-struct', 'data')
save('test_timetable_steps_v73.mat', '-struct', 'data', '-v7.3')
