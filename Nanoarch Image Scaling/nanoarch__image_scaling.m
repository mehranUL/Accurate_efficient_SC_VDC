% -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
% _    _ _        _               ______  __     ________ _______ _______ ______
%| |  | | |      | |        /\   |  ____/\\ \   / /  ____|__   __|__   __|  ____|
%| |  | | |      | |       /  \  | |__ /  \\ \_/ /| |__     | |     | |  | |__
%| |  | | |      | |      / /\ \ |  __/ /\ \\   / |  __|    | |     | |  |  __|
%| |__| | |____  | |____ / ____ \| | / ____ \| |  | |____   | |     | |  | |____
% \____/|______| |______/_/    \_\_|/_/    \_\_|  |______|  |_|     |_|  |______|
% -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

% The code was partially referenced: https://stackoverflow.com/questions/26142288/resize-an-image-with-bilinear-interpolation-without-imresize
 
% UL Lafayette, Research Sc. Dr. Sercan Aygun
% The supervisor Asst. Prof. Dr. M. Hassan Najafi
% version-1.1 05/10/2023

%Bilinear Interpolation - Deterministic and Stochastic

clear all
close all

scaling = 2; % Upscaling to bigger image
N = 2^8; %Bitstream Size
SC_scale = N/256; 


% NON-NOISY - DETERMINISTIC
im = rgb2gray(imread('pepper.png')); 
figure
imshow(im);
title("Noissless original");

out_dims = size(im);
out_dims = out_dims * scaling;

%// Get some necessary variables first
in_rows = size(im,1);
in_cols = size(im,2);
out_rows = out_dims(1);
out_cols = out_dims(2);

%// Let S_R = R / R'
S_R = in_rows / out_rows;
%// Let S_C = C / C'
S_C = in_cols / out_cols;

%// Define grid of co-ordinates in our image
%// Generate (x,y) pairs for each point in our image
[cf, rf] = meshgrid(1 : out_cols, 1 : out_rows);

%// Let r_f = r'*S_R for r = 1,...,R'
%// Let c_f = c'*S_C for c = 1,...,C'
rf = rf * S_R;
cf = cf * S_C;

%// Let r = floor(rf) and c = floor(cf)
r = floor(rf);
c = floor(cf);

%// Any values out of range, cap
r(r < 1) = 1;
c(c < 1) = 1;
r(r > in_rows - 1) = in_rows - 1;
c(c > in_cols - 1) = in_cols - 1;

%// Let delta_R = rf - r and delta_C = cf - c
delta_R = rf - r;
delta_C = cf - c;

%// Final line of algorithm
%// Get column major indices for each point we wish
%// to access
in1_ind = sub2ind([in_rows, in_cols], r, c);
in2_ind = sub2ind([in_rows, in_cols], r+1,c);
in3_ind = sub2ind([in_rows, in_cols], r, c+1);
in4_ind = sub2ind([in_rows, in_cols], r+1, c+1);

%// Now interpolate
%// Go through each channel for the case of colour
%// Create output image that is the same class as input
out_nonnoisy = zeros(out_rows, out_cols, size(im, 3));
out_nonnoisy = cast(out_nonnoisy, class(im));

chan = double(im(:,:,1)); %// Get i'th channel
%// Interpolate the channel

% tmp = chan(in1_ind).*(1 - delta_R).*(1 - delta_C) + ...
%     chan(in2_ind).*(delta_R).*(1 - delta_C) + ...
%     chan(in3_ind).*(1 - delta_R).*(delta_C) + ...
%     chan(in4_ind).*(delta_R).*(delta_C);
% out_nonnoisy(:,:,1) = cast(tmp, class(im));

tic
% for row = 1:1:out_dims(1)
%     for column = 1:1:out_dims(2)
% out(row, column) = chan(in1_ind(row, column))*(1 - delta_R(row, column))*(1 - delta_C(row, column)) + ...
%     chan(in2_ind(row, column))*(delta_R(row, column))*(1 - delta_C(row, column)) + ...
%     chan(in3_ind(row, column))*(1 - delta_R(row, column))*(delta_C(row, column)) + ...
%     chan(in4_ind(row, column))*(delta_R(row, column))*(delta_C(row, column));
%     end
% end

for row = 1:1:out_dims(1)
    for column = 1:1:out_dims(2)
        out(row, column) = deterministic_filter(chan(in1_ind(row, column)), chan(in2_ind(row, column)), chan(in3_ind(row, column)), chan(in4_ind(row, column)), delta_R(row, column), delta_C(row, column));
    end
end

det_time = toc;

out_nonnoisy(:,:,1) = cast(out, class(im));

deterministic_nonoise = out_nonnoisy;

figure
imshow(out_nonnoisy);
title("Processed noiseless deterministic");

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

% NON-NOISY - STOCHASTIC

im = double(rgb2gray(imread('pepper.png'))); 

im = SC_scale.*im;

out_dims = size(im);
out_dims = out_dims * scaling;

%// Get some necessary variables first
in_rows = size(im,1);
in_cols = size(im,2);
out_rows = out_dims(1);
out_cols = out_dims(2);

%// Let S_R = R / R'
S_R = in_rows / out_rows;
%// Let S_C = C / C'
S_C = in_cols / out_cols;

%// Define grid of co-ordinates in our image
%// Generate (x,y) pairs for each point in our image
[cf, rf] = meshgrid(1 : out_cols, 1 : out_rows);

%// Let r_f = r'*S_R for r = 1,...,R'
%// Let c_f = c'*S_C for c = 1,...,C'
rf = rf * S_R;
cf = cf * S_C;

%// Let r = floor(rf) and c = floor(cf)
r = floor(rf);
c = floor(cf);

%// Any values out of range, cap
r(r < 1) = 1;
c(c < 1) = 1;
r(r > in_rows - 1) = in_rows - 1;
c(c > in_cols - 1) = in_cols - 1;

%// Let delta_R = rf - r and delta_C = cf - c
delta_R = rf - r;
delta_C = cf - c;

%// Final line of algorithm
%// Get column major indices for each point we wish
%// to access
in1_ind = sub2ind([in_rows, in_cols], r, c);
in2_ind = sub2ind([in_rows, in_cols], r+1,c);
in3_ind = sub2ind([in_rows, in_cols], r, c+1);
in4_ind = sub2ind([in_rows, in_cols], r+1, c+1);

%// Now interpolate
%// Go through each channel for the case of colour
%// Create output image that is the same class as input
out = zeros(out_rows, out_cols, size(im, 3));
out = cast(out, class(im));

chan = double(im(:,:,1)); %already a single channel

%sequences = transpose(net(sobolset(6),(N)));
%sequences = faure((N-1),6,10);
%sequences = weyl_6(N);
%sequences = VonDerCorput_6(N);
%sequences = Hammersley_Sercan(P,N);
sequences = niederreiter2_generate(20, N, 2, 31);

%sequences = LFSR_Bulk(N)/N;

%sequences = transpose(mohsenvanderdata);

for row = 1:1:out_dims(1)
    for column = 1:1:out_dims(2)
        in1 = number_source(chan(in1_ind(row, column))/256, N, sequences(1, :));
        in2 = number_source(chan(in2_ind(row, column))/256, N, sequences(1, :));
        in3 = number_source(chan(in3_ind(row, column))/256, N, sequences(1, :));
        in4 = number_source(chan(in4_ind(row, column))/256, N, sequences(1, :));
        s1 = number_source(delta_R(row, column), N, sequences(6, :)); 
        s2 = number_source(delta_C(row, column), N, sequences(6, :));
        
        tmp = mux4_to_1_adder(in1, in2, in3, in4, s1, s2);
        out(row, column) = sum(tmp);
    end
end

out = out./SC_scale;
SC_nonnoisy = uint8(out);

figure 
imshow(SC_nonnoisy);
%title("Processed noiseless SC");



psnr_value = psnr(deterministic_nonoise, SC_nonnoisy)
ssim_value = ssim(deterministic_nonoise, SC_nonnoisy)







