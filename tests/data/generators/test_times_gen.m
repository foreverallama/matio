%% Datetime

dt_basic = datetime(2025, 4, 1, 12, 00, 00);
dt_tz = datetime(2025,4,1,12,00,00, 'TimeZone', "America/New_York");

dt_vector = datetime(2025, 4, 1) + days(0:5);
dt_array = reshape(dt_array, 2, 3);

dt_empty = datetime.empty;
dt_fmt = datetime(2025,4,1,12,00,00, 'Format', 'yyyy-MM-dd HH:mm:ss');

data = struct;
data.dt_basic = dt_basic;
data.dt_vector = dt_vector;
data.dt_array = dt_array;
data.dt_empty = dt_empty;
data.dt_tz = dt_tz;
data.dt_fmt = dt_fmt;

%% Duration

dur_s = seconds(5);
dur_m = minutes(5);
dur_h = hours(5);
dur_D = days(5);
dur_hms = duration(1,2,3); % 1h, 2m, 3s
dur_array = seconds([10, 20, 30; 40, 50, 60]);
dur_empty = duration.empty;
dur_Y = years([1 2 3]);

data.dur_s = dur_s;
data.dur_m = dur_m;
data.dur_h = dur_h;
data.dur_days = dur_D;
data.dur_hms = dur_hms;
data.dur_array = dur_array;
data.dur_empty = dur_empty;
data.dur_years = dur_Y;

%% calendarDuration

cdur_empty = calendarDuration.empty(0, 0);
cdur_days = caldays([1 2 3]);
cdur_weeks = calweeks([1 2]);
cdur_days_and_months = caldays([1 2]) + calmonths([1 0]);
cdur_months_and_years = calyears(1) + calmonths([0 6]);
cdur_days_and_qtrs = calquarters(1) + caldays(15);
cdur_array = [calmonths(1), caldays(5); calmonths(2), caldays(10)];
cdur_millis = caldays(1) + duration(1, 2, 3);  % 1 day + 1h 2m 3s

data.cdur_empty = cdur_empty;
data.cdur_days = cdur_days;
data.cdur_weeks = cdur_weeks;
data.cdur_days_and_months = cdur_days_and_months;
data.cdur_months_and_years = cdur_months_and_years;
data.cdur_days_and_qtrs = cdur_days_and_qtrs;
data.cdur_array = cdur_array;
data.cdur_millis = cdur_millis;

%% Saving

save('test_time_v7.mat', '-struct', 'data');
save('test_time_v73.mat', '-struct', 'data', '-v7.3');
