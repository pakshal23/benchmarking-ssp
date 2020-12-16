function [H, M] = get_measurement_matrix(experiment, param, K)
%--------------------------------------------------------------------------
% Inputs: experiment: 'deconv_gaussian', 'deconv_airy_disk', 'fourier_samp'
%         param: For 'deconv_gaussian' -> variance of the Gaussian filter
%                For 'deconv_airy_disk' -> filter_size (all the other params are fixed
%                from GlobalBioIm 2D example)
%                For 'fourier_samp' -> Number of measurements
%         K: Length of the signal
%
% Outputs: H:  [M x K] system matrix
%          M: Number of measurements
%--------------------------------------------------------------------------

switch experiment
    
    case 'deconv_gaussian'
        
        % We keep only those rows of the convolution matrix where the
        % entire filter is included.
        
        std_dev = sqrt(param);
        filter_size = 2*round(3*std_dev)+1;
        h = fspecial('gaussian', [filter_size,1], std_dev);
        H_full = convmtx(h, K);
        H = H_full(filter_size:K,:);    
        M = size(H, 1);
    
    case 'deconv_airy_disk'
        
        % We keep only those rows of the convolution matrix where the
        % entire filter is included.
        
        filter_size = param;
        hfs = (filter_size-1)/2;
        
        lamb=561;                % Illumination wavelength
        res=30;                  % Resolution (nm)
        Na=1.4;                  % Objective numerical aperture
        fc=2*Na/lamb*res;        % cut-off frequency
        
        ll=linspace(-0.5, 0, K/2+1);
        lr=linspace(0, 0.5, K/2);
        distance_grid = abs([ll,lr(2:end)]');
        otf = fftshift(1/pi*(2*acos(abs(distance_grid)/fc)-sin(2*acos(abs(distance_grid)/fc))).*(distance_grid<fc));
        psf = real(ifft2(otf));
        h = ifftshift(psf);
        h = h(K/2+1-hfs:K/2+1+hfs);                  % Truncating the psf
        H_full = convmtx(h, K);
        H = H_full(filter_size:K,:);    
        M = size(H, 1);
        
    case 'fourier_samp'    

        M = param;
        H = zeros([M, K]);
        F = dftmtx(K);
        FR = real(F);
        FI = imag(F);
        
        ind1 = 2:4;                                                        % Fixed low frequency indices
        ind = [ind1];

        indices_lf = 5:10;                                                 % Low frequency indices (to sample from randomly)

        if ((M-1)/2-3 >= 3)
            ind2 = indices_lf(randperm(length(indices_lf),3));
            ind = [ind ind2];
        end

        indices_hf = 11:K/2;                                               % High frequency indices (to sample from randomly)

        if ((M-1)/2 - 6 > 0)
            ind3 = indices_hf(randperm(length(indices_hf),(M-1)/2 - 6));
            ind = [ind ind3];
        end
        
        H(1,:) = FR(1,:);

        k = 1;
        for i = 2:2:(M-1)
            H(i,:) = FR(ind(k),:);
            H(i+1,:) = FI(ind(k),:);
            k = k + 1;
        end
        
end

end

