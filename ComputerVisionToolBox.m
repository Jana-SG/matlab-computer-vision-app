classdef ProgrammingTask3 < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                       matlab.ui.Figure
        GridLayout                     matlab.ui.container.GridLayout
        TabGroup                       matlab.ui.container.TabGroup
        imageenhancementTab            matlab.ui.container.Tab
        darkenSlider                   matlab.ui.control.Slider
        darkenSliderLabel              matlab.ui.control.Label
        histogramButton                matlab.ui.control.Button
        brightenSlider                 matlab.ui.control.Slider
        brightenSliderLabel            matlab.ui.control.Label
        UIAxes_HistN                   matlab.ui.control.UIAxes
        UIAxes_HistO                   matlab.ui.control.UIAxes
        SpatialdomainfilteringTab      matlab.ui.container.Tab
        SharpenPanel                   matlab.ui.container.Panel
        PrewittButton                  matlab.ui.control.Button
        SobelPanel                     matlab.ui.container.Panel
        boostingAlphaEditField_2       matlab.ui.control.NumericEditField
        boostingAlphaEditField_2Label  matlab.ui.control.Label
        verticalButton                 matlab.ui.control.Button
        horizontalButton               matlab.ui.control.Button
        LaplacianPanel                 matlab.ui.container.Panel
        boostingAlphaEditField         matlab.ui.control.NumericEditField
        boostingAlphaEditFieldLabel    matlab.ui.control.Label
        secondderivativeButton         matlab.ui.control.Button
        firstderivativeButton          matlab.ui.control.Button
        SmoothPanel                    matlab.ui.container.Panel
        filterSizeSpinner              matlab.ui.control.Spinner
        filterSizeSpinnerLabel         matlab.ui.control.Label
        MedianCheckBox                 matlab.ui.control.CheckBox
        WeightedAverageCheckBox        matlab.ui.control.CheckBox
        BoxFilterCheckBox              matlab.ui.control.CheckBox
        FrequnecydomainfilteringTab    matlab.ui.container.Tab
        SharpenPanel_2                 matlab.ui.container.Panel
        cutofffrequencySpinner_4       matlab.ui.control.Spinner
        cutofffrequencySpinner_4Label  matlab.ui.control.Label
        cutofffrequencySpinner_2       matlab.ui.control.Spinner
        cutofffrequencySpinner_2Label  matlab.ui.control.Label
        orderSpinner                   matlab.ui.control.Spinner
        orderSpinnerLabel              matlab.ui.control.Label
        GaussianHighpassfilteringCheckBox  matlab.ui.control.CheckBox
        IdealButterworthfilteringHPFCheckBox  matlab.ui.control.CheckBox
        SmoothPanel_2                  matlab.ui.container.Panel
        cutofffrequencySpinner_3       matlab.ui.control.Spinner
        cutofffrequencySpinner_3Label  matlab.ui.control.Label
        cutofffrequencySpinner         matlab.ui.control.Spinner
        cutofffrequencySpinnerLabel    matlab.ui.control.Label
        OrderSpinner                   matlab.ui.control.Spinner
        OrderSpinnerLabel              matlab.ui.control.Label
        GaussianlowpassfilteringCheckBox  matlab.ui.control.CheckBox
        IdealButterworthfilteringLPFCheckBox  matlab.ui.control.CheckBox
        ColorSpaceTab                  matlab.ui.container.Tab
        YCbCrCheckBox                  matlab.ui.control.CheckBox
        LabCheckBox                    matlab.ui.control.CheckBox
        HSICheckBox                    matlab.ui.control.CheckBox
        RGBCheckBox                    matlab.ui.control.CheckBox
        UIAxes3                        matlab.ui.control.UIAxes
        UIAxes2                        matlab.ui.control.UIAxes
        UIAxes                         matlab.ui.control.UIAxes
        PyramidsFiltersTab             matlab.ui.container.Tab
        FilterBanksPanel               matlab.ui.container.Panel
        ApplyselectedfilterButton      matlab.ui.control.Button
        ScaleDropDown                  matlab.ui.control.DropDown
        ScaleDropDownLabel             matlab.ui.control.Label
        OrientationDropDown            matlab.ui.control.DropDown
        EdgeFiltersLabel               matlab.ui.control.Label
        FilterTypeDropDown             matlab.ui.control.DropDown
        FilterTypeDropDownLabel        matlab.ui.control.Label
        TemplateMatcingPanel           matlab.ui.container.Panel
        GuassianPyramidsCheckBox       matlab.ui.control.CheckBox
        NormalizedcrosscorrelationCheckBox  matlab.ui.control.CheckBox
        SumSquareDifferenceCheckBox    matlab.ui.control.CheckBox
        ZeromeanCorrelationCheckBox    matlab.ui.control.CheckBox
        SelecttemplatefromimageButton  matlab.ui.control.Button
        LoadTemplateButton             matlab.ui.control.Button
        PyramidsPanel                  matlab.ui.container.Panel
        LevelSpinnerLabel              matlab.ui.control.Label
        LevelSpinner                   matlab.ui.control.Spinner
        levelSpinner                   matlab.ui.control.Spinner
        levelSpinnerLabel              matlab.ui.control.Label
        LaplacianPyramidCheckBox       matlab.ui.control.CheckBox
        ReductiontimesSpinner          matlab.ui.control.Spinner
        ReductiontimesSpinnerLabel     matlab.ui.control.Label
        GuassianPyramidCheckBox        matlab.ui.control.CheckBox
        UIAxesTemplate                 matlab.ui.control.UIAxes
        edgeandcornerTab               matlab.ui.container.Tab
        CornerDetectionPanel           matlab.ui.container.Panel
        SelectSpecificRegionButton     matlab.ui.control.Button
        numberofCornersSpinner         matlab.ui.control.Spinner
        numberofCornersSpinnerLabel    matlab.ui.control.Label
        neighborhoodsizeSpinner        matlab.ui.control.Spinner
        neighborhoodsizeSpinnerLabel   matlab.ui.control.Label
        MinCornerStrengthSpinner       matlab.ui.control.Spinner
        MinCornerStrengthSpinnerLabel  matlab.ui.control.Label
        HarrisCornerDetectionButton    matlab.ui.control.Button
        EdgedetectionPanel             matlab.ui.container.Panel
        SigmaSlider                    matlab.ui.control.Slider
        SigmaSliderLabel               matlab.ui.control.Label
        thresholdSlider                matlab.ui.control.RangeSlider
        thresholdSliderLabel           matlab.ui.control.Label
        CannyEdgedetectionButton       matlab.ui.control.Button
        HOGHoughTab                    matlab.ui.container.Tab
        HoughPanel                     matlab.ui.container.Panel
        MinLengthEditField             matlab.ui.control.NumericEditField
        MinLengthEditFieldLabel        matlab.ui.control.Label
        fillGapEditField               matlab.ui.control.NumericEditField
        fillGapEditFieldLabel          matlab.ui.control.Label
        maxRadiusEditField             matlab.ui.control.NumericEditField
        maxRadiusEditFieldLabel        matlab.ui.control.Label
        minRadiusEditField             matlab.ui.control.NumericEditField
        minRadiusEditFieldLabel        matlab.ui.control.Label
        CircleHoughwithpolarityCheckBox  matlab.ui.control.CheckBox
        polarityDropDown               matlab.ui.control.DropDown
        polarityDropDownLabel          matlab.ui.control.Label
        ThresholdSpinner               matlab.ui.control.Spinner
        ThresholdSpinnerLabel          matlab.ui.control.Label
        CircleHoughTransformCheckBox   matlab.ui.control.CheckBox
        numPeaksEditField              matlab.ui.control.NumericEditField
        numPeaksEditFieldLabel         matlab.ui.control.Label
        LineHoughTransformCheckBox     matlab.ui.control.CheckBox
        HOGPanel                       matlab.ui.container.Panel
        CellSizeEditField              matlab.ui.control.NumericEditField
        CellSizeEditFieldLabel         matlab.ui.control.Label
        BlockSizeEditField             matlab.ui.control.NumericEditField
        BlockSizeEditFieldLabel        matlab.ui.control.Label
        BinNumberEditField             matlab.ui.control.NumericEditField
        BinNumberEditFieldLabel        matlab.ui.control.Label
        HOGCheckBox                    matlab.ui.control.CheckBox
        DoGLoGPanel                    matlab.ui.container.Panel
        sizeSpinner                    matlab.ui.control.Spinner
        sizeSpinnerLabel               matlab.ui.control.Label
        SigmaSpinner                   matlab.ui.control.Spinner
        SigmaSpinnerLabel              matlab.ui.control.Label
        SizeSpinner                    matlab.ui.control.Spinner
        SizeSpinnerLabel               matlab.ui.control.Label
        LoGCheckBox                    matlab.ui.control.CheckBox
        sigma2Spinner                  matlab.ui.control.Spinner
        sigma2Label                    matlab.ui.control.Label
        sigma1Spinner                  matlab.ui.control.Spinner
        sigma1Label                    matlab.ui.control.Label
        DoGCheckBox                    matlab.ui.control.CheckBox
        UIAxesHist                     matlab.ui.control.UIAxes
        RANSACTab                      matlab.ui.container.Tab
        distancethresholdSpinner       matlab.ui.control.Spinner
        distancethresholdSpinnerLabel  matlab.ui.control.Label
        samplenumberSpinner            matlab.ui.control.Spinner
        samplenumberSpinnerLabel       matlab.ui.control.Label
        CircledetectionButton          matlab.ui.control.Button
        LinedetectionButton            matlab.ui.control.Button
        DistancethresholdSpinner       matlab.ui.control.Spinner
        DistancethresholdSpinnerLabel  matlab.ui.control.Label
        ConfidenceSpinner              matlab.ui.control.Spinner
        ConfidenceSpinnerLabel         matlab.ui.control.Label
        NumberofSamplesSpinner         matlab.ui.control.Spinner
        NumberofSamplesSpinnerLabel    matlab.ui.control.Label
        MatchFeaturesDropDown_2        matlab.ui.control.DropDown
        MatchFeaturesDropDown_2Label   matlab.ui.control.Label
        RANSACButton                   matlab.ui.control.Button
        LoadsecondimageButton          matlab.ui.control.Button
        UIAxesransac                   matlab.ui.control.UIAxes
        StereoVisionTab                matlab.ui.container.Tab
        StereoVisionPanel              matlab.ui.container.Panel
        MatchFeaturesDropDown          matlab.ui.control.DropDown
        MatchFeaturesDropDownLabel     matlab.ui.control.Label
        EpipolarlinesIMG2Button        matlab.ui.control.Button
        EpipolarlinesIMG1Button        matlab.ui.control.Button
        ApplyRANSACButton              matlab.ui.control.Button
        LoadSecondImageButton          matlab.ui.control.Button
        UIAxesMatches                  matlab.ui.control.UIAxes
        BasIcOperationsPanel           matlab.ui.container.Panel
        RESETButton                    matlab.ui.control.Button
        BlackandWhiteButton            matlab.ui.control.Button
        GreyscaleButton                matlab.ui.control.Button
        loadimageButton                matlab.ui.control.Button
        UIAxesImage                    matlab.ui.control.UIAxes
        UIAxesModify                   matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        modIMG
        IMG
        IMG_G
        filter_size  
        Sb_x
        Alpha
        Alpha_S
        orderlps
        orderhps
        Dlps
        Dhps
        DGL
        DGH
        pyraN
        templateIMG
        lmFilters
        threshold
        gaussianPyramid
        laplacianPyramid
        CG
        Lthresh
        Hthresh
        MinQuality
        HfilterSize
        numCorner
        HarrisROI
        maxLevels
        modIMG2
        IMG2
        matchedPoints1
        matchedPoints2
        inliers
        fMatrix
    end
    
    
    methods (Access = private)
        
        function F = makeLMfilters(app)
        SUP = 49;
        SCALEX = sqrt(2).^[1:3];
        NORIENT = 6;

        NROTINV = 12;
        NBAR = length(SCALEX)*NORIENT;
        NEDGE = length(SCALEX)*NORIENT;
        NF = NBAR + NEDGE + NROTINV;
        F = zeros(SUP, SUP, NF);
        hsup = (SUP - 1)/2;
        [x, y] = meshgrid(-hsup:hsup, hsup:-1:-hsup);
        orgpts = [x(:) y(:)]';

        count = 1;
        for scale = 1:length(SCALEX)
            for orient = 0:NORIENT-1
                angle = pi * orient / NORIENT;
                c = cos(angle); s = sin(angle);
                rotpts = [c -s; s c] * orgpts;
                F(:,:,count) = app.makefilter(SCALEX(scale), 0, 1, rotpts, SUP);
                F(:,:,count + NEDGE) = app.makefilter(SCALEX(scale), 0, 2, rotpts, SUP);
                count = count + 1;
            end
        end

        count = NBAR + NEDGE + 1;
        SCALES = sqrt(2).^[1:4];
        for i = 1:length(SCALES)
            F(:,:,count)   = app.normalise(fspecial('gaussian', SUP, SCALES(i)));
            F(:,:,count+1) = app.normalise(fspecial('log', SUP, SCALES(i)));
            F(:,:,count+2) = app.normalise(fspecial('log', SUP, 3*SCALES(i)));
            count = count + 3;
        end
    end

    function f = makefilter(app, scale, phasex, phasey, pts, sup)
        gx = app.gauss1d(3*scale, 0, pts(1,:), phasex);
        gy = app.gauss1d(scale, 0, pts(2,:), phasey);
        f = app.normalise(reshape(gx .* gy, sup, sup));
    end

    function g = gauss1d(~, sigma, mean, x, ord)
        x = x - mean;
        variance = sigma^2;
        denom = 2 * variance;
        g = exp(-x.^2 / denom) / sqrt(pi * denom);
        switch ord
            case 1
                g = -g .* (x / variance);
            case 2
                g = g .* ((x.^2 - variance) / variance^2);
        end
    end

    function f = normalise(~, f)
        f = f - mean(f(:));
        f = f / sum(abs(f(:)));
    end
    

    function [bestLine, inlierIdx] = ransacLine(app,points, threshold, numIterations)
        numPoints = size(points, 1);
        bestInlierCount = 0;
        bestLine = [];

        for i = 1:numIterations
            % Randomly pick 2 points
            idx = randperm(numPoints, 2);
            p1 = points(idx(1), :);
            p2 = points(idx(2), :);

            % Skip if identical
            if isequal(p1, p2)
                continue;
            end

            % Line model: Ax + By + C = 0
            A = p2(2) - p1(2);
            B = p1(1) - p2(1);
            C = p2(1)*p1(2) - p1(1)*p2(2);

            % Distance from all points to line
            dists = abs(A*points(:,1) + B*points(:,2) + C) / sqrt(A^2 + B^2);

            % Inliers
            inlierIdx = find(dists < threshold);
            if numel(inlierIdx) > bestInlierCount
                bestInlierCount = numel(inlierIdx);
                bestLine = [A, B, C];
                bestInliers = inlierIdx;
            end
        end

        inlierIdx = bestInliers;
    end
    function [center, radius, inlierIdx] = ransacCircle(app,points, threshold, numIterations)
    numPoints = size(points,1);
    bestInlierCount = 0;

    for i = 1:numIterations
        % Randomly pick 3 points
        idx = randperm(numPoints, 3);
        pts = points(idx, :);

        % Fit circle from 3 points
        [c, r] = fitCircleFrom3Points(app,pts);

        if isnan(r) || r == 0
            continue;
        end

        % Compute distances to center
        dists = sqrt(sum((points - c).^2, 2));
        inlierIdx = find(abs(dists - r) < threshold);

        if numel(inlierIdx) > bestInlierCount
            bestInlierCount = numel(inlierIdx);
            center = c;
            radius = r;
            bestInliers = inlierIdx;
        end
    end

    inlierIdx = bestInliers;
end

    function [center, radius] = fitCircleFrom3Points(app,pts)
        A = 2 * (pts(2,:) - pts(1,:));
        B = 2 * (pts(3,:) - pts(1,:));
        C = sum(pts(2,:).^2 - pts(1,:).^2);
        D = sum(pts(3,:).^2 - pts(1,:).^2);
    
        M = [A; B];
        rhs = [C; D];

        if rank(M) < 2
            center = [NaN NaN];
            radius = NaN;
            return;
        end

        center = (M \ rhs)';
        radius = norm(center - pts(1,:));
    end

    function mask = createCircularMask(app,imageSize, center, radius)
        [xx, yy] = meshgrid(1:imageSize(2), 1:imageSize(1));
        dist = sqrt((xx - center(1)).^2 + (yy - center(2)).^2);
        mask = dist <= radius;
    end
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: loadimageButton
        function loadimageButtonPushed(app, event)
         [imageName, pathName] = uigetfile('*.*',"Pick an Image");
         imageName = strcat(pathName,imageName);
         app.IMG = imread(imageName);
         app.modIMG = app.IMG;
         imshow(app.IMG,'Parent',app.UIAxesImage)
         app.TabGroup.Visible = "on";
         app.GreyscaleButton.Enable = "on";
         app.BlackandWhiteButton.Enable = 'on';
         app.RESETButton.Enable = "on";
         app.lmFilters = makeLMfilters(app);

         imgSize = size(app.IMG);
         minDim = min(imgSize(1), imgSize(2));
         app.maxLevels = floor(log2(minDim)) - 1;
         app.ReductiontimesSpinner.Limits = [1 app.maxLevels];
        end

        % Button pushed function: histogramButton
        function histogramButtonPushed(app, event)
            if size(app.modIMG, 3) == 1
                
                app.IMG_G = app.modIMG;
                eqimg = histeq(app.IMG_G);
                h_o = imhist(app.IMG_G);
                h_n = imhist(eqimg);
            else
                hsv_img = rgb2hsv(app.modIMG);
                V_original = hsv_img(:,:,3);
                h_o = imhist(im2uint8(V_original)); 
                hsv_img(:,:,3) = histeq(hsv_img(:,:,3));
                eqimg = hsv2rgb(hsv_img);
                V_equalized = hsv_img(:,:,3);
                h_n = imhist(im2uint8(V_equalized));
            end

            plot(h_o,'Parent',app.UIAxes_HistO);
            plot(h_n,'Parent',app.UIAxes_HistN);
            imshow(eqimg,'Parent',app.UIAxesModify);
            title(app.UIAxesModify, 'Histogram equalized image');

            %app.modIMG = uint8(eqimg);
            app.modIMG = im2uint8(mat2gray(eqimg));


        end

        % Value changed function: brightenSlider
        function brightenSliderValueChanged(app, event)
            value = app.brightenSlider.Value;
            brightImg = uint8(double(app.modIMG) * value);
            imshow(brightImg,'Parent',app.UIAxesModify);
            app.modIMG = brightImg;
            title(app.UIAxesModify, 'Brightened image');
        end

        % Value changed function: darkenSlider
        function darkenSliderValueChanged(app, event)
            value = app.darkenSlider.Value;
            darkenimg = uint8(double(app.modIMG) * value); 
            imshow(darkenimg,'Parent',app.UIAxesModify);
            app.modIMG = darkenimg;
            title(app.UIAxesModify, 'Darkened image');
        end

        % Value changed function: HSICheckBox
        function HSICheckBoxValueChanged(app, event)
            value = app.HSICheckBox.Value;
           if value 
            HSI = rgb2hsv(app.modIMG);
            H = HSI(:,:,1);
            S = HSI(:,:,2);
            I = HSI(:,:,3);
            imshow(H,"Parent",app.UIAxes);
            title(app.UIAxes, 'Hue Channel');
            imshow(S,"Parent",app.UIAxes2);
            title(app.UIAxes2, 'Saturation Channel');
            imshow(I,"Parent",app.UIAxes3);
            title(app.UIAxes3, 'Intensity Channel');
            imshow(HSI,"Parent",app.UIAxesModify)
            title(app.UIAxesModify, 'Full HSI Image');
            app.modIMG = HSI;
           else
                cla(app.UIAxes);
                cla(app.UIAxes2);
                cla(app.UIAxes3);
                cla(app.UIAxesModify);
           end 
        end

        % Button pushed function: GreyscaleButton
        function GreyscaleButtonPushed(app, event)
            grayscale = im2gray(app.modIMG);
            imshow(grayscale,"Parent",app.UIAxesModify);
            title(app.UIAxesModify, 'Greyscale image');
            app.modIMG = grayscale;
        end

        % Button pushed function: BlackandWhiteButton
        function BlackandWhiteButtonPushed(app, event)
            gray = im2gray(app.modIMG);
            BW = imbinarize(gray);
            imshow(BW,"Parent",app.UIAxesModify);
            title(app.UIAxesModify, 'Binary Image');
            app.modIMG = BW;
        end

        % Value changed function: LabCheckBox
        function LabCheckBoxValueChanged(app, event)
            value = app.LabCheckBox.Value;
            if value 
            LAB = rgb2lab(app.modIMG);
            L = LAB(:,:,1);
            A = LAB(:,:,2);
            B = LAB(:,:,3);
            L_normalized = L/100;
            A_normalized = (A + 128)/255;
            B_normalized = (B + 128)/255;
            imshow(L_normalized,"Parent",app.UIAxes);
            title(app.UIAxes, 'Lightness Channel');
            imshow(A_normalized,"Parent",app.UIAxes2);
            title(app.UIAxes2, 'chrominance A Channel');
            imshow(B_normalized,"Parent",app.UIAxes3);
            title(app.UIAxes3, 'chrominance B Channel');
            imshow(LAB,"Parent",app.UIAxesModify)
            title(app.UIAxesModify, 'Full L*a*b* Image');
            app.modIMG = LAB;
            else
                cla(app.UIAxes);
                cla(app.UIAxes2);
                cla(app.UIAxes3);
                cla(app.UIAxesModify);
           end
        end

        % Value changed function: YCbCrCheckBox
        function YCbCrCheckBoxValueChanged(app, event)
            value = app.YCbCrCheckBox.Value;
            if value 
            ycbcr = rgb2ycbcr(app.modIMG);
            luma = ycbcr(:,:,1);
            cb = ycbcr(:,:,2);
            cr = ycbcr(:,:,3);

            imshow(luma,"Parent",app.UIAxes);
            title(app.UIAxes, 'Luma Channel');

            imshow(cb,"Parent",app.UIAxes2);
            title(app.UIAxes2, 'chrominance Blue Channel');
            imshow(cr,"Parent",app.UIAxes3);
            title(app.UIAxes3, 'chrominance Red Channel');
            imshow(ycbcr,"Parent",app.UIAxesModify)
            title(app.UIAxesModify, 'Full YCbCr Image');
            app.modIMG = ycbcr;
            
        else
                cla(app.UIAxes);
                cla(app.UIAxes2);
                cla(app.UIAxes3);
                cla(app.UIAxesModify);
            end
        end

        % Button pushed function: RESETButton
        function RESETButtonPushed(app, event)
         app.modIMG = app.IMG;
        imshow(app.IMG,'Parent',app.UIAxesModify);
        title(app.UIAxesModify, 'Reset to Original');
        end

        % Value changed function: RGBCheckBox
        function RGBCheckBoxValueChanged(app, event)
            value = app.RGBCheckBox.Value;
            if value 
            RGB = app.modIMG;
            R = RGB(:,:,1);
            G = RGB(:,:,2);
            B = RGB(:,:,3);
            imshow(R,"Parent",app.UIAxes);
            title(app.UIAxes, 'Red');
            imshow(G,"Parent",app.UIAxes2);
            title(app.UIAxes2, 'Green');
            imshow(B,"Parent",app.UIAxes3);
            title(app.UIAxes3, 'Blue');
            imshow(RGB,"Parent",app.UIAxesModify);
            title(app.UIAxesModify, 'Full RGB Image');
            app.modIMG = RGB;
            else
                cla(app.UIAxes);
                cla(app.UIAxes2);
                cla(app.UIAxes3);
                cla(app.UIAxesModify);
            end
        end

        % Value changed function: filterSizeSpinner
        function filterSizeSpinnerValueChanged(app, event)
            value = app.filterSizeSpinner.Value;
            app.filter_size = value;
            app.WeightedAverageCheckBox.Enable = 'on';
            app.MedianCheckBox.Enable = 'on';
            app.BoxFilterCheckBox.Enable = "on";
        end

        % Value changed function: BoxFilterCheckBox
        function BoxFilterCheckBoxValueChanged(app, event)
            value = app.BoxFilterCheckBox.Value;
            if value 
                 app.modIMG = double(app.modIMG); 
                if app.filter_size == 3
                    H_box_3 = (1/9).*[1 1 1; 1 1 1; 1 1 1];
                    filtered = imfilter(double(app.modIMG),H_box_3);
                    imshow(uint8(filtered),"Parent",app.UIAxesModify);
                    title(app.UIAxesModify, 'Filtered image');
                else 
                    H_box = (1/25).* ones(app.filter_size,app.filter_size);
                    filtered = imfilter(double(app.modIMG),H_box);
                    imshow(uint8(filtered),"Parent",app.UIAxesModify);
                    title(app.UIAxesModify, "Filtered image");
                end
                app.modIMG = uint8(filtered);
            
            end 
        end

        % Value changed function: WeightedAverageCheckBox
        function WeightedAverageCheckBoxValueChanged(app, event)
            value = app.WeightedAverageCheckBox.Value;
            if value
                app.modIMG = double(app.modIMG); 
                switch app.filter_size 
                    case 3
                        H_avg_3 = (1/16).*[1 2 1; 2 4 2; 1 2 1];
                        filter = imfilter(double(app.modIMG),H_avg_3);
                        imshow(uint8(filter),"Parent",app.UIAxesModify);
                        title(app.UIAxesModify, "Filtered image");
                    case 5
                        H_avg_5 = (1/65).*[1 2 2 2 1; 
                                           1 2 4 2 1;
                                           2 4 8 4 2;
                                           1 2 4 2 1;
                                           1 2 2 2 1];
                        filter = imfilter(double(app.modIMG),H_avg_5);
                        imshow(uint8(filter),"Parent",app.UIAxesModify);
                        title(app.UIAxesModify, "Filtered image");
                    case 7
                        H_avg_7 = (1/128).*[1 1 1 2 1 1 1; 
                                            1 1 2 4 1 1 2;
                                            1 2 4 8 4 2 1;
                                            2 4 8 16 8 4 2;
                                            1 2 4 8 4 2 1;
                                            1 1 2 4 1 1 2;
                                            1 1 1 2 1 1 1];
                        filter = imfilter(double(app.modIMG),H_avg_7);
                        imshow(uint8(filter),"Parent",app.UIAxesModify);
                        title(app.UIAxesModify, "Filtered image");
                    case 9
                         H_avg_9 = (1/280).*[1 1 1 1 2 1 1 1 1;
                                             1 1 1 2 4 2 1 1 1; 
                                             1 1 2 4 8 4 2 1 1;
                                             1 2 4 8 16 8 4 2 1;
                                             2 4 8 16 32 16 8 4 2;
                                             1 2 4 8 16 8 4 2 1;
                                             1 1 2 4 8 4 2 1 1;
                                             1 1 1 2 4 2 1 1 1;
                                             1 1 1 1 2 1 1 1 1];
                         filter = imfilter(double(app.modIMG),H_avg_9);
                         imshow(uint8(filter),"Parent",app.UIAxesModify);
                         title(app.UIAxesModify, "Filtered image");
                end 
                 app.modIMG = uint8(filter);
            
            end
        end

        % Value changed function: MedianCheckBox
        function MedianCheckBoxValueChanged(app, event)
            value = app.MedianCheckBox.Value;
            if value
                switch app.filter_size
                    case 3
                        filter_s = [3 3];
                    case 5
                        filter_s = [5 5];
                    case 7
                        filter_s = [7 7];
                    case 9
                        filter_s = [9 9];
                end

                if size(app.modIMG, 3) == 1
                    filtered = medfilt2(double(app.modIMG), filter_s);
                else
                    filtered = zeros(size(app.modIMG));
                    for channel = 1:3
                        filtered(:,:,channel) = medfilt2(double(app.modIMG(:,:,channel)), filter_s);
                    end
                end
                imshow(uint8(filtered), "Parent", app.UIAxesModify);
                title(app.UIAxesModify, "Filtered image");
                app.modIMG = uint8(filtered);
            end 
        end

        % Button pushed function: firstderivativeButton
        function firstderivativeButtonPushed(app, event)
            if isempty(app.Alpha) || ~isnumeric(app.Alpha)
                 app.Alpha = 0; 
            end
            app.modIMG = double(app.modIMG);
            Lp_1 = [0 -1 0; -1 4+app.Alpha -1; 0 -1 0];
            filter_Lp = imfilter(double(app.modIMG),Lp_1);
            filter_Lp = app.modIMG+filter_Lp;
            imshow(uint8(filter_Lp),"Parent",app.UIAxesModify);
            title(app.UIAxesModify, "Filtered image");
            app.modIMG = uint8(filter_Lp);
        end

        % Button pushed function: secondderivativeButton
        function secondderivativeButtonPushed(app, event)
            if isempty(app.Alpha) || ~isnumeric(app.Alpha)
                 app.Alpha = 0; 
            end
           
            app.modIMG = double(app.modIMG);
            Lp_2 = [-1 -1 -1; -1 8+app.Alpha -1; -1 -1 -1];
            filter_Lp = imfilter(double(app.modIMG),Lp_2);
            filter_Lp = app.modIMG+filter_Lp;
            imshow(uint8(filter_Lp),"Parent",app.UIAxesModify);
            title(app.UIAxesModify, "Filtered image");
            app.modIMG = uint8(filter_Lp);
        end

        % Button pushed function: horizontalButton
        function horizontalButtonPushed(app, event)
            if isempty(app.Alpha_S) || ~isnumeric(app.Alpha_S)
                 app.Alpha_S = 0; 
            end

            app.modIMG = double(app.modIMG);
            app.Sb_x = [-1 -2-app.Alpha_S -1; 0 0 0; 1 2+app.Alpha_S 1];
            sobel_H = imfilter(double(app.modIMG),app.Sb_x);
            sobel_H = app.modIMG + sobel_H;
            imshow(uint8(sobel_H),"Parent",app.UIAxesModify);
            title(app.UIAxesModify, "Filtered image");
            app.modIMG = uint8(sobel_H);
        end

        % Button pushed function: verticalButton
        function verticalButtonPushed(app, event)
            if isempty(app.Alpha_S) || ~isnumeric(app.Alpha_S)
                 app.Alpha_S = 0; 
            end

            app.modIMG = double(app.modIMG);
            Sb_y = app.Sb_x';
            sobel_V = imfilter(double(app.modIMG),Sb_y);
            sobel_V = app.modIMG+sobel_V;
            imshow(uint8(sobel_V),"Parent",app.UIAxesModify);
            title(app.UIAxesModify, "Filtered image");
            app.modIMG = uint8(sobel_V);
        end

        % Value changed function: boostingAlphaEditField
        function boostingAlphaEditFieldValueChanged(app, event)
            value = app.boostingAlphaEditField.Value;
            app.Alpha = value;
        end

        % Value changed function: boostingAlphaEditField_2
        function boostingAlphaEditField_2ValueChanged(app, event)
            value = app.boostingAlphaEditField_2.Value;
            app.Alpha_S = value;
        end

        % Value changed function: GaussianlowpassfilteringCheckBox
        function GaussianlowpassfilteringCheckBoxValueChanged(app, event)
            value = app.GaussianlowpassfilteringCheckBox.Value;
            if value 
                [FI, D] = prepFFT(app,app.modIMG);
                if isempty(app.DGL) 
                   app.DGL = 30; 
                end
                GLPF = exp(-D.^2/(2*app.DGL^2));
                if iscell(FI)
                    filtered_img = zeros(size(app.modIMG));
                    for channel = 1:3
                        Gfilter_channel = ifft2(fftshift(FI{channel}.*GLPF));
                        filtered_img(:,:,channel) = abs(Gfilter_channel);
                    end
                else

                    Gfilter = ifft2(fftshift(FI.*GLPF));
                    filtered_img = abs(Gfilter);
                end
                imshow(uint8(filtered_img),'Parent',app.UIAxesModify)
                title(app.UIAxesModify, "Filtered image");
                app.modIMG = uint8(filtered_img);
            end
        end

        % Value changed function: IdealButterworthfilteringLPFCheckBox
        function IdealButterworthfilteringLPFCheckBoxValueChanged(app, event)
            value = app.IdealButterworthfilteringLPFCheckBox.Value;
            if value
                 [FI, D] = prepFFT(app,app.modIMG);
                if isempty(app.orderlps) 
                 app.orderlps = 1; 
                end 
                if isempty(app.Dlps)
                     app.Dlps = 10; 
                end
                BLPF = 1./(1.0 + (D./ app.Dlps).^(2*app.orderlps));

                if iscell(FI)
                    filtered_img = zeros(size(app.modIMG));
                    for channel = 1:3
                        filter_channel = ifft2(fftshift(FI{channel}.*BLPF));
                        filtered_img(:,:,channel) = abs(filter_channel);
                    end
                else
                    filter = ifft2(fftshift(FI.*BLPF));
                    filtered_img = abs(filter);
                end
                imshow(uint8(filtered_img),'Parent',app.UIAxesModify)
                title(app.UIAxesModify, "Filtered image");
                app.modIMG = uint8(filtered_img);
            end 
        end

        % Value changed function: OrderSpinner
        function OrderSpinnerValueChanged(app, event)
            value = app.OrderSpinner.Value;
            app.orderlps = value;
        end

        % Value changed function: cutofffrequencySpinner
        function cutofffrequencySpinnerValueChanged(app, event)
            value = app.cutofffrequencySpinner.Value;
            app.Dlps = value;
        end

        % Value changed function: orderSpinner
        function orderSpinnerValueChanged(app, event)
            value = app.orderSpinner.Value;
            app.orderhps = value;
        end

        % Value changed function: cutofffrequencySpinner_2
        function cutofffrequencySpinner_2ValueChanged(app, event)
            value = app.cutofffrequencySpinner_2.Value;
            app.Dhps = value;
        end

        % Value changed function: IdealButterworthfilteringHPFCheckBox
        function IdealButterworthfilteringHPFCheckBoxValueChanged(app, event)
            value = app.IdealButterworthfilteringHPFCheckBox.Value;
            if value
                 [FI, D] = prepFFT(app,app.modIMG);
                if isempty(app.orderhps) 
                 app.orderhps = 1; 
                end 
                if isempty(app.Dhps)
                     app.Dhps = 10; 
                end
                BHPF = 1./ (1.0+(app.Dhps./ D).^(2*app.orderhps));
                
                if iscell(FI)
                    filtered_img = zeros(size(app.modIMG));
                    for channel = 1:3
                        filter_channel = ifft2(fftshift(FI{channel}.*BHPF));
                        filtered_img(:,:,channel) = abs(filter_channel);
                    end
                else
                   filter = ifft2(fftshift(FI.*BHPF));
                   filtered_img = abs(filter);
                end
                
                imshow(uint8(filtered_img),'Parent',app.UIAxesModify)
                title(app.UIAxesModify, "Filtered image");
                app.modIMG = uint8(filtered_img);
            end 
        end

        % Value changed function: cutofffrequencySpinner_3
        function cutofffrequencySpinner_3ValueChanged(app, event)
            value = app.cutofffrequencySpinner_3.Value;
            app.DGL = value;
        end

        % Value changed function: cutofffrequencySpinner_4
        function cutofffrequencySpinner_4ValueChanged(app, event)
            value = app.cutofffrequencySpinner_4.Value;
            app.DGH = value;
        end

        % Value changed function: GaussianHighpassfilteringCheckBox
        function GaussianHighpassfilteringCheckBoxValueChanged(app, event)
            value = app.GaussianHighpassfilteringCheckBox.Value;
            if value 
                [FI, D] = prepFFT(app,app.modIMG);
                if isempty(app.DGH) 
                   app.DGH = 30; 
                end
                GHPF = 1-exp(-D.^2/(2*app.DGH^2));

                if iscell(FI)
                    filtered_img = zeros(size(app.modIMG));
                    for channel = 1:3
                        Gfilter_channel = ifft2(fftshift(FI{channel}.*GHPF));
                        filtered_img(:,:,channel) = abs(Gfilter_channel);
                    end
                else
                     Gfilter = ifft2(fftshift(FI.*GHPF));
                     filtered_img = abs(Gfilter);
                end
                
                imshow(uint8(filtered_img),'Parent',app.UIAxesModify)
                title(app.UIAxesModify, "Filtered image");
                app.modIMG = uint8(filtered_img);
            end
        end

        % Button pushed function: PrewittButton
        function PrewittButtonPushed(app, event)
            img = double(app.modIMG); 
             p_msk = [-1 0 1; -1 0 1; -1 0 1];

            if size(img, 3) == 3  % RGB image
                 k_combined = zeros(size(img));
                 for c = 1:3
                    kx = conv2(img(:,:,c), p_msk, 'same');
                    ky = conv2(img(:,:,c), p_msk', 'same');
                    k_combined(:,:,c) = sqrt(kx.^2 + ky.^2);
                end
                edged_img = uint8(k_combined);
            else  
                kx = conv2(img, p_msk, 'same');
                ky = conv2(img, p_msk', 'same');
                edged_img = uint8(sqrt(kx.^2 + ky.^2));
            end

            imshow(edged_img, 'Parent', app.UIAxesModify)
            title(app.UIAxesModify, "Filtered image");
            app.modIMG = edged_img;

        end

        % Button pushed function: LoadTemplateButton
        function LoadTemplateButtonPushed(app, event)
            [imageName, pathName] = uigetfile('*.*',"Pick an Image");
            imageName = strcat(pathName,imageName);
            app.templateIMG = imread(imageName);
            app.templateIMG = im2gray(app.templateIMG);
        end

        % Button pushed function: SelecttemplatefromimageButton
        function SelecttemplatefromimageButtonPushed(app, event)
            figure;
            imshow(app.modIMG);
            title('Draw a rectangle to select the template region');

            rect = getrect; 
            Template = imcrop(app.modIMG, rect);

            if isempty(Template)
                uialert(app.UIFigure, 'No region selected', 'Selection Error');
            return;
            end
            app.templateIMG = im2gray(Template);
            close(gcf);  
        end

        % Value changed function: ZeromeanCorrelationCheckBox
        function ZeromeanCorrelationCheckBoxValueChanged(app, event)
            value = app.ZeromeanCorrelationCheckBox.Value;
            if value
                img = double(im2gray(app.modIMG));             
                template = double(app.templateIMG);           
                template = template - mean(template(:));       

                [h, w] = size(template);
                [H, W] = size(img);

                zmc_map = zeros(H - h + 1, W - w + 1);

                for i = 1:(H - h + 1)
                    for j = 1:(W - w + 1)
                        region = img(i:i+h-1, j:j+w-1);         
                        zmc_map(i, j) = sum(template(:) .* region(:)); 
                    end
                end

                responseMap = mat2gray(zmc_map);

               
                [~, idx] = max(responseMap(:));
                [y, x] = ind2sub(size(responseMap), idx);

                imshow(responseMap, 'Parent', app.UIAxesTemplate);
                title(app.UIAxesTemplate, 'Zero-Mean Cross Correlation');

                imshow(app.modIMG, [], 'Parent', app.UIAxesModify);
                title(app.UIAxesModify, 'ZMC Best Match');
                drawrectangle(app.UIAxesModify, 'Position', [x, y, w, h], ...
                        'EdgeColor', 'b', 'LineWidth', 2, 'FaceAlpha', 0);
    
            end
            
        end

        % Value changed function: SumSquareDifferenceCheckBox
        function SumSquareDifferenceCheckBoxValueChanged(app, event)
            value = app.SumSquareDifferenceCheckBox.Value;
            if value 
                
                img = double(im2gray(app.modIMG));
                template = double(app.templateIMG);
                
                [h, w] = size(template);
                [H, W] = size(img);

                ssd_map = zeros(H - h + 1, W - w + 1);

                for i = 1:(H - h + 1)
                    for j = 1:(W - w + 1)
                        region = img(i:i+h-1, j:j+w-1);
                        diff = region - template;
                        ssd_map(i, j) = sum(diff(:).^2);
                    end
                end

                responseMap = max(ssd_map(:)) - ssd_map;  
                responseMap = mat2gray(responseMap);      
                
                [~, idx] = max(responseMap(:));
                [y, x] = ind2sub(size(responseMap), idx);
                
                imshow(app.modIMG, [] ,'Parent', app.UIAxesModify);
                title(app.UIAxesModify, 'SSD Best Match');
                hold(app.UIAxesModify, 'on');
                rectangle(app.UIAxesModify, 'Position', [x, y, w, h], ...
                  'EdgeColor', 'b','LineWidth', 2,'FaceAlpha',0);
                hold(app.UIAxesModify, 'off');

                imshow(responseMap, 'Parent', app.UIAxesTemplate);
                title(app.UIAxesTemplate, 'SSD Response Map ');

                
                %app.modIMG = responseMap;
            end
        end

        % Value changed function: NormalizedcrosscorrelationCheckBox
        function NormalizedcrosscorrelationCheckBoxValueChanged(app, event)
            value = app.NormalizedcrosscorrelationCheckBox.Value;
            if value 
                img = im2gray(app.modIMG);
                template = app.templateIMG;

                c = normxcorr2(template, img);
                [ypeak,xpeak] = find(c== max(c(:)));

                yoffSet = ypeak - size(template,1) ;
                xoffSet = xpeak - size(template,2) ;
                
                imshow(c, [], 'Parent', app.UIAxesTemplate);
                title(app.UIAxesTemplate, 'Normalized Cross Correlation');
                
                imshow(app.modIMG, [], 'Parent', app.UIAxesModify);
                title(app.UIAxesModify, 'Detected Matches');
                
                drawrectangle(app.UIAxesModify,'Position',[xoffSet,yoffSet,size(template,2),size(template,1)], ...
                 'FaceAlpha',0);
                
                %app.modIMG = c;
            end 
        end

        % Value changed function: GuassianPyramidCheckBox
        function GuassianPyramidCheckBoxValueChanged(app, event)
            value = app.GuassianPyramidCheckBox.Value;
            if value
                pyramid = app.modIMG;
                if isempty(app.pyraN)
                    app.pyraN = 1;
                end
        
                app.gaussianPyramid = cell(1, app.pyraN + 1);
                app.gaussianPyramid{1} = pyramid;  
        
                pyramidDouble = im2double(pyramid);
        
               for i = 1:app.pyraN
                    pyramidDouble = impyramid(pyramidDouble, 'reduce');
                    app.gaussianPyramid{i+1} = im2uint8(pyramidDouble);  
                end
        
                imshow(app.gaussianPyramid{app.pyraN + 1},[], 'Parent', app.UIAxesModify);
                app.modIMG = app.gaussianPyramid{app.pyraN + 1};
           end
    
            app.LevelSpinner.Enable = "on";
            app.LevelSpinner.Limits = [1, app.pyraN + 1];
            
            

        end

        % Value changed function: ReductiontimesSpinner
        function ReductiontimesSpinnerValueChanged(app, event)
            value = app.ReductiontimesSpinner.Value;
            app.pyraN = value;
            app.LaplacianPyramidCheckBox.Enable = "on";
            app.GuassianPyramidCheckBox.Enable = "on";
        end

        % Value changed function: LaplacianPyramidCheckBox
        function LaplacianPyramidCheckBoxValueChanged(app, event)
            value = app.LaplacianPyramidCheckBox.Value;
            if value
                img = app.modIMG;
                if isempty(app.pyraN)
                    app.pyraN = 1;
                end
        
                isColorImage = (size(img, 3) == 3);
                %Guassian Pyramid 
                G = cell(app.pyraN + 1, 1);
                G{1} = im2double(img);  
                
                for i = 2:(app.pyraN + 1)
                    if isColorImage
                       reducedImg = zeros(ceil(size(G{i-1},1)/2), ceil(size(G{i-1},2)/2), 3);
                        for c = 1:3
                            channelImg = G{i-1}(:,:,c);
                            reducedImg(:,:,c) = impyramid(channelImg, 'reduce');
                        end
                        G{i} = reducedImg;
                    else
                        G{i} = impyramid(G{i-1}, 'reduce');
                    end
                end

                % Laplacian Pyramid
                L = cell(app.pyraN, 1);
                for i = 1:app.pyraN
                    if isColorImage
                        upsampledImg = zeros(size(G{i}));
                        for c = 1:3
                            smallerImg = G{i+1}(:,:,c);
                            expanded = impyramid(smallerImg, 'expand');
                    
                            expanded = imresize(expanded, [size(G{i},1), size(G{i},2)]);
                
                            upsampledImg(:,:,c) = expanded;
                            L{i}(:,:,c) = G{i}(:,:,c) - upsampledImg(:,:,c);

                        end
                    else
                        expanded = impyramid(G{i+1}, 'expand');
                        expanded = imresize(expanded, [size(G{i},1), size(G{i},2)]);
                        L{i} = G{i} - expanded;
                    end
                end

                app.laplacianPyramid = L;
                app.levelSpinner.Limits = [1, length(L)];
        
                if isColorImage
                    laplacianTop = zeros(size(L{1}));
                    for c = 1:3
                       channelData = L{1}(:,:,c);
                        if max(channelData(:)) == min(channelData(:))
                            laplacianTop(:,:,c) = zeros(size(channelData)); 
                        else
                            laplacianTop(:,:,c) = mat2gray(channelData);
                        end
                    end
                else
                    if max(L{1}(:)) == min(L{1}(:))
                        laplacianTop = zeros(size(L{1})); 
                    else
                        laplacianTop = mat2gray(L{1});
                    end
                end

                laplacianTop = im2uint8(laplacianTop);
                imshow(laplacianTop, 'Parent', app.UIAxesModify);
                title(app.UIAxesModify, 'Laplacian Pyramid - Level 1');
                app.levelSpinner.Enable = "on";

                app.modIMG = laplacianTop;
            else
                app.levelSpinner.Enable = 'off';
            end
        
        end

        % Value changed function: levelSpinner
        function levelSpinnerValueChanged(app, event)
            value = app.levelSpinner.Value;
            img = app.modIMG;
            isColorImage = (size(img, 3) == 3);
            %color Image
            if isColorImage
                selectedLevelImage = zeros(size(app.laplacianPyramid{value}));
                for c = 1:3
                    channelData = app.laplacianPyramid{value}(:,:,c);
                    if max(channelData(:)) == min(channelData(:))
                        selectedLevelImage(:,:,c) = zeros(size(channelData)); 
                    else
                        selectedLevelImage(:,:,c) = mat2gray(channelData);
                    end
                end
            else
                %greyscale Image
                if max(app.laplacianPyramid{value}(:)) == min(app.laplacianPyramid{value}(:))
                    selectedLevelImage = zeros(size(app.laplacianPyramid{value})); 
                else
                    selectedLevelImage = mat2gray(app.laplacianPyramid{value});
                end
            end
            selectedLevelImage = im2uint8(selectedLevelImage);
            imshow(selectedLevelImage, 'Parent', app.UIAxesModify);
            title(app.UIAxesModify, sprintf('Laplacian Pyramid - Level %d', value));
            app.modIMG = selectedLevelImage;
        end

        % Value changed function: thresholdSlider
        function thresholdSliderValueChanged(app, event)
            value = app.thresholdSlider.Value;
                app.Lthresh = value(1);
                app.Hthresh = value(2);
            
        end

        % Button pushed function: CannyEdgedetectionButton
        function CannyEdgedetectionButtonPushed(app, event)
            if isempty(app.CG)
                app.CG = 1.0;
            end
            if isempty(app.Lthresh)
                app.Lthresh = 0.1;
            end
            if isempty(app.Hthresh)
                app.Hthresh = 0.3;
            end
            if size(app.modIMG, 3) == 3
                edgeR = edge(app.modIMG(:,:,1), 'canny', [app.Lthresh app.Hthresh], app.CG);
                edgeG = edge(app.modIMG(:,:,2), 'canny', [app.Lthresh app.Hthresh], app.CG);
                edgeB = edge(app.modIMG(:,:,3), 'canny', [app.Lthresh app.Hthresh], app.CG);
        
                Fulledge = edgeR | edgeG | edgeB;
       
            else
                Fulledge = edge(app.modIMG, 'canny', [app.Lthresh app.Hthresh], app.CG);
            end
            
            imshow(Fulledge, 'Parent', app.UIAxesModify);
            title(app.UIAxesModify, 'Canny Edge Detection');
            app.modIMG = Fulledge;

            app.CG = [];
            app.Lthresh = [];
            app.Hthresh = [];

        end

        % Value changed function: SigmaSlider
        function SigmaSliderValueChanged(app, event)
            value = app.SigmaSlider.Value;
            if value 
                app.CG = value;
            end
        end

        % Button pushed function: HarrisCornerDetectionButton
        function HarrisCornerDetectionButtonPushed(app, event)
            if isempty(app.MinQuality)
                app.MinQuality = 0.03;
            end
            if isempty(app.HfilterSize)
                app.HfilterSize = 5;
            end
            if isempty(app.numCorner)
                app.numCorner = 200;
            end

            if ~isempty(app.HarrisROI)  
                ROI = round(app.HarrisROI);  
                imgROI = imcrop(app.modIMG, ROI);
            else
                imgROI = app.modIMG;
                ROI = [];
            end

            
            if size(imgROI, 3) == 3
                cornersR = detectHarrisFeatures(imgROI(:,:,1), 'MinQuality', app.MinQuality, 'FilterSize', app.HfilterSize);
                cornersG = detectHarrisFeatures(imgROI(:,:,2), 'MinQuality', app.MinQuality, 'FilterSize', app.HfilterSize);
                cornersB = detectHarrisFeatures(imgROI(:,:,3), 'MinQuality', app.MinQuality, 'FilterSize', app.HfilterSize);

                allCorners = [cornersR.Location; cornersG.Location; cornersB.Location];

                if ~isempty(allCorners)
                    [~, uniqueIdx] = uniquetol(allCorners, 0.01, 'ByRows', true);
                    uniqueCorners = allCorners(uniqueIdx, :);
                    corners = cornerPoints(uniqueCorners);
                else
                    corners = cornerPoints.empty;
                end
            else
                corners = detectHarrisFeatures(imgROI, 'MinQuality', app.MinQuality, 'FilterSize', app.HfilterSize);
            end

            if ~isempty(ROI)
                rectangle(app.UIAxesModify, 'Position', ROI, 'EdgeColor', 'g', 'LineWidth', 1.5);
               
                offset = ROI(1:2);
                corners.Location = corners.Location + offset;
            end

            imshow(app.modIMG, 'Parent', app.UIAxesModify);
            hold(app.UIAxesModify, 'on');
            strongestCorners = corners.selectStrongest(app.numCorner);
            plot(app.UIAxesModify, strongestCorners.Location(:,1), strongestCorners.Location(:,2), 'r+');
            hold(app.UIAxesModify, 'off');
            title(app.UIAxesModify, 'Harris Corner Detection');
        end

        % Value changed function: MinCornerStrengthSpinner
        function MinCornerStrengthSpinnerValueChanged(app, event)
            value = app.MinCornerStrengthSpinner.Value;
            if value 
                app.MinQuality  = value;
            end
        end

        % Value changed function: neighborhoodsizeSpinner
        function neighborhoodsizeSpinnerValueChanged(app, event)
            value = app.neighborhoodsizeSpinner.Value;
            if value 
                app.HfilterSize = value;
            end
        end

        % Value changed function: numberofCornersSpinner
        function numberofCornersSpinnerValueChanged(app, event)
            value = app.numberofCornersSpinner.Value;
            if value
                 app.numCorner = value;
            end
        end

        % Button pushed function: SelectSpecificRegionButton
        function SelectSpecificRegionButtonPushed(app, event)
            imshow(app.modIMG, 'Parent', app.UIAxesModify);
            r = drawrectangle(app.UIAxesModify, 'Color', 'g');
            app.HarrisROI = round(r.Position);
        end

        % Value changed function: GuassianPyramidsCheckBox
        function GuassianPyramidsCheckBoxValueChanged(app, event)
            value = app.GuassianPyramidsCheckBox.Value;
            
            if value
                img = double(im2gray(app.modIMG));
                template = double(im2gray(app.templateIMG));
                
                %max level for search
                minDim = min(size(template));
                maxUsableLevel = floor(log2(minDim)) - 1; 
                usedMaxLevels = min(app.maxLevels, maxUsableLevel);
        
                pyramid{1} = img;
                for lvl = 2:usedMaxLevels
                    pyramid{lvl} = impyramid(pyramid{lvl-1}, 'reduce');
                end
        
                bestScore = -Inf;
                bestX = 0;
                bestY = 0;
                bestW = 0;
                bestH = 0;
                bestLevel = 1;
        
                
                for lvl = 1:usedMaxLevels
                    imgLvl = pyramid{lvl};
                    scale = size(img, 1) / size(imgLvl, 1);
            
                    
                    reTemplate = imresize(template, 1/scale);
                    if any(size(reTemplate) > size(imgLvl))
                        continue;
                    end
            
            
                    c = normxcorr2(reTemplate, imgLvl);
                    [peakY, peakX] = find(c == max(c(:)));
                    score = c(peakY, peakX);
            
                    if score > bestScore
                        bestScore = score;
                        bestX = round((peakX - size(reTemplate, 2)) * scale);
                        bestY = round((peakY - size(reTemplate, 1)) * scale);
                        bestW = round(size(reTemplate, 2) * scale);
                        bestH = round(size(reTemplate, 1) * scale);
                        bestLevel = lvl;
                    end
                end
        
       
                imshow(app.modIMG, [], 'Parent', app.UIAxesModify);
                title(app.UIAxesModify, sprintf('Best Match at Level %d', bestLevel));
                drawrectangle(app.UIAxesModify, ...
                    'Position', [bestX, bestY, bestW, bestH], ...
                    'EdgeColor', 'b', 'LineWidth', 2, 'FaceAlpha', 0);
            end
        end

        % Value changed function: FilterTypeDropDown
        function FilterTypeDropDownValueChanged(app, event)
            value = app.FilterTypeDropDown.Value;
            app.ScaleDropDown.Items = {'1', '2', '3', '4'};
            app.ScaleDropDown.Enable = 'on';

            switch value
                case 'LoG (sigma)'
                    app.OrientationDropDown.Enable = 'off';
                    app.ScaleDropDown.Items = {'1', '2', '3', '4'};
                case 'LoG (3 sigma)'
                    app.OrientationDropDown.Enable = 'off';
                    app.ScaleDropDown.Items = {'1', '2', '3', '4'};
                case 'Gaussian'
                    app.OrientationDropDown.Enable = 'off';
                    app.ScaleDropDown.Items = {'1', '2', '3', '4'};
                otherwise
                    app.ScaleDropDown.Items = {'1', '2','3'};
                    app.OrientationDropDown.Enable = 'on';
            end
        end

        % Button pushed function: ApplyselectedfilterButton
        function ApplyselectedfilterButtonPushed(app, event)
            filterType = app.FilterTypeDropDown.Value;
            orientation = app.OrientationDropDown.Value;
            scale = str2double(app.ScaleDropDown.Value);
                
            switch orientation
                case '0 degrees'
                    orientIdx = 0;
                case '30 degrees'
                    orientIdx = 1;
                case '60 degrees'
                    orientIdx = 2;
                case '90 degrees'
                    orientIdx = 3;
                case '120 degrees'
                    orientIdx = 4;
                case '150 degrees'
                    orientIdx = 5;
                otherwise
                    orientIdx = 0;
             end
    
            filterIdx = 0;
            switch filterType
                case 'Bar '
                    % 1-18
                    filterIdx = (scale-1)*6 + orientIdx + 1;
            
                case 'Edge'
                    % 19-36
                    filterIdx = 18 + (scale-1)*6 + orientIdx + 1;
            
                case 'Gaussian'
                    %  37, 40, 43, 46
                    filterIdx = 36 + (scale-1)*3 + 1;
            
                case 'LoG (sigma)'
                    % 38, 41, 44, 47
                    filterIdx = 36 + (scale-1)*3 + 2;
                case 'LoG (3 sigma)'
                     % 39, 42, 45, 48
                    filterIdx = 36 + (scale-1)*3 + 3;
            end
            
            selectedFilter = app.lmFilters(:, :, filterIdx);
            img = im2double(app.modIMG);
            isColor = (size(img, 3) == 3);

            filtered_img = zeros(size(img));
            energyMap = zeros(size(img, 1), size(img, 2));
    
            if isColor
                % color 
                for c = 1:3
                    response = imfilter(img(:,:,c), selectedFilter, 'replicate');
                    energyMap = energyMap + response .^ 2;
                    filtered_img(:,:,c) =  img(:,:,c) + response;
                end
                
                filtered_img = min(max(filtered_img, 0), 1);  

            else
                % grayscale 
                response = imfilter(img, selectedFilter, 'replicate');
                energyMap = response .^ 2;
                filtered_img = mat2gray(img + response);
                
            end
    

            
            energyMap = sqrt(energyMap);
            energyMap = mat2gray(energyMap);
    
            
            imshow(energyMap, 'Parent', app.UIAxesModify);
            title(app.UIAxesModify,"Energy map");
    
            imshow(im2uint8(filtered_img), 'Parent', app.UIAxesTemplate);
            title(app.UIAxesTemplate,"Filtered Image");
            app.modIMG = im2uint8(filtered_img);
        end

        % Value changed function: LevelSpinner
        function LevelSpinnerValueChanged(app, event)
            value = app.LevelSpinner.Value;
            selectedLevelImage = app.gaussianPyramid{value};
            imshow(selectedLevelImage, 'Parent', app.UIAxesModify);
            title(app.UIAxesModify, sprintf('Gaussian Pyramid - Level %d', value));
            app.modIMG = selectedLevelImage;
       
        end

        % Value changed function: DoGCheckBox
        function DoGCheckBoxValueChanged(app, event)
            value = app.DoGCheckBox.Value;
            if value
                sigma1 = app.sigma1Spinner.Value;
                sigma2 = app.sigma2Spinner.Value;
                size = app.sizeSpinner.Value;
                if sigma1 >= sigma2
                    uialert(app.UIFigure, ...
                        'Sigma 1 must be smaller than Sigma 2.', ...
                        'Invalid Input');
                    app.DoGCheckBox.Value = false;
                    return;
                end

                grayIMG = im2gray(app.modIMG);
                gaussian1 = fspecial('Gaussian', size, sigma1);
                gaussian2 = fspecial('Gaussian', size, sigma2);
                dog = gaussian1 - gaussian2;
                dogFilterImage = conv2(double(grayIMG), dog, 'same');
                imshow(mat2gray(dogFilterImage), 'Parent', app.UIAxesModify);
                title(app.UIAxesModify,"DoG filtered Image");
                app.modIMG = mat2gray(dogFilterImage);
            end
        end

        % Value changed function: LoGCheckBox
        function LoGCheckBoxValueChanged(app, event)
            value = app.LoGCheckBox.Value;
            if value 
                Sigma = app.SigmaSpinner.Value;
                Size = app.SizeSpinner.Value;

                log = fspecial('log', Size, Sigma);
                logFilterImage = imfilter(app.modIMG,log);
                imshow(mat2gray(logFilterImage), 'Parent', app.UIAxesModify);
                title(app.UIAxesModify,"LoG filtered Image");
                app.modIMG = mat2gray(logFilterImage);
            end

        end

        % Value changed function: HOGCheckBox
        function HOGCheckBoxValueChanged(app, event)
            value = app.HOGCheckBox.Value;
            if value 
                cellSize = app.CellSizeEditField.Value; 
                blockSize = app.BlockSizeEditField.Value;
                Binnumber = app.BinNumberEditField.Value;
                
                [featureVector,hogVisualization] = extractHOGFeatures(app.modIMG,...
                    'CellSize', [cellSize cellSize ], ...
                    'BlockSize', [blockSize blockSize], ...
                    'NumBins', Binnumber);
                imshow(app.modIMG, 'Parent', app.UIAxesModify);
                hold(app.UIAxesModify, 'on');
                plot(hogVisualization, app.UIAxesModify);
                hold(app.UIAxesModify, 'off');
                title(app.UIAxesModify, "HOG Features");
                
                bar(app.UIAxesHist, featureVector);
                title(app.UIAxesHist, 'HOG Feature Vector Histogram');
            end 
        end

        % Value changed function: LineHoughTransformCheckBox
        function LineHoughTransformCheckBoxValueChanged(app, event)
            value = app.LineHoughTransformCheckBox.Value;
            if value
                peaks = app.numPeaksEditField.Value;
                FillGap = app.fillGapEditField.Value;
                MinLength = app.MinLengthEditField.Value;
                gray = im2gray(app.modIMG);
                edges = edge(gray,'canny');
                [H,theta,rho] = hough(edges,'Theta', -90:0.5:89);
                imshow(imadjust(rescale(H)), ...
                         'XData', theta,...
                         'YData', rho,...
                         'InitialMagnification', 'fit',...
                         'Parent', app.UIAxesHist);
                hold(app.UIAxesHist, 'on'); 

                xlabel(app.UIAxesHist, '\theta (degrees)');
                ylabel(app.UIAxesHist, '\rho (pixels)');
                title(app.UIAxesHist, 'Hough Transform Accumulator');
                axis(app.UIAxesHist, 'on');
                axis(app.UIAxesHist, 'normal')

                peaks = houghpeaks(H, peaks);
                plot(app.UIAxesHist, theta(peaks(:, 2)), rho(peaks(:, 1)), ...
                    'rs', 'LineWidth', 1.5, 'MarkerSize', 6);
                hold(app.UIAxesHist, 'off');

                lines = houghlines(edges, theta, rho, peaks,'FillGap',FillGap,'MinLength',MinLength);
                imshow(app.modIMG, 'Parent', app.UIAxesModify);
                hold(app.UIAxesModify, 'on');
        
        
                for k = 1:length(lines)
                    xy = [lines(k).point1; lines(k).point2];
                    plot(app.UIAxesModify, xy(:,1), xy(:,2), ...
                        'LineWidth', 2, 'Color', 'green');
                end
        
                title(app.UIAxesModify, 'Detected Lines');
                hold(app.UIAxesModify, 'off');
            end
        end

        % Value changed function: CircleHoughTransformCheckBox
        function CircleHoughTransformCheckBoxValueChanged(app, event)
            value = app.CircleHoughTransformCheckBox.Value;
            if value
                minRadius = app.minRadiusEditField.Value;
                maxRadius = app.maxRadiusEditField.Value;
                radiiRange = [minRadius maxRadius];
                thresh = app.ThresholdSpinner.Value;
                gray = im2gray(app.modIMG);
                gray = edge(gray, 'Canny');
                [centers, radii] = imfindcircles(gray,radiiRange,...
                                    "EdgeThreshold",thresh);
                imshow(app.modIMG, 'Parent', app.UIAxesModify);
                title(app.UIAxesModify,'Circles Detected');

                hold(app.UIAxesModify, 'on');
                viscircles(app.UIAxesModify, centers, radii, 'EdgeColor', 'b');
                plot(app.UIAxesModify, centers(:,1), centers(:,2), 'r+', 'MarkerSize', 8, 'LineWidth', 1);
                hold(app.UIAxesModify, 'off');
            end
        end

        % Value changed function: CircleHoughwithpolarityCheckBox
        function CircleHoughwithpolarityCheckBoxValueChanged(app, event)
            value = app.CircleHoughwithpolarityCheckBox.Value;
            if value
                minRadius = app.minRadiusEditField.Value;
                maxRadius = app.maxRadiusEditField.Value;
                radiiRange = [minRadius maxRadius];
                thresh = app.ThresholdSpinner.Value;
                polarity = app.polarityDropDown.Value;
                gray = im2gray(app.modIMG);
                
                [centers, radii] = imfindcircles(gray,radiiRange,...
                                    "EdgeThreshold",thresh,...
                                    'ObjectPolarity', polarity);
                imshow(app.modIMG, 'Parent', app.UIAxesModify);
                title(app.UIAxesModify,'Circles Detected using imfindcircles');

                hold(app.UIAxesModify, 'on');
                viscircles(app.UIAxesModify, centers, radii, 'EdgeColor', 'b');
                plot(app.UIAxesModify, centers(:,1), centers(:,2), 'r+', 'MarkerSize', 8, 'LineWidth', 1);
                hold(app.UIAxesModify, 'off');
            end
        end

        % Button pushed function: LoadSecondImageButton
        function LoadSecondImageButtonPushed(app, event)
            [imageName, pathName] = uigetfile('*.*',"Pick an Image");
            imageName = strcat(pathName,imageName);
            app.IMG2 = imread(imageName);
            app.modIMG2 = app.IMG2;
            imshow(app.IMG2,'Parent',app.UIAxesModify)
            title(app.UIAxesModify,"Second Image")
        end

        % Button pushed function: ApplyRANSACButton
        function ApplyRANSACButtonPushed(app, event)
            [app.fMatrix, app.inliers] = estimateFundamentalMatrix(...
                    app.matchedPoints1, app.matchedPoints2, ...
                    'Method', 'RANSAC','NumTrials', 4000);
            if app.MatchFeaturesDropDown.Value == "precomputed"
                imshow(app.modIMG, 'Parent', app.UIAxesImage);
                hold(app.UIAxesImage, 'on');
               
                pts1 = app.matchedPoints1(app.inliers, :);
                plot(app.UIAxesImage, pts1(:,1), pts1(:,2), 'go');
                
                title(app.UIAxesImage, 'Inliers in First Image');
                hold(app.UIAxesImage, 'off');
                imshow(app.modIMG2, 'Parent', app.UIAxesModify);
                hold(app.UIAxesModify, 'on');

                pts2 = app.matchedPoints2(app.inliers, :);
                    plot(app.UIAxesModify, pts2(:,1), pts2(:,2), 'go');
                title(app.UIAxesModify, 'Inliers in Second Image');
                hold(app.UIAxesModify, 'off');
            else
                imshow(app.modIMG, 'Parent', app.UIAxesImage);
                hold(app.UIAxesImage, 'on');
               
                pts1 = app.matchedPoints1(app.inliers).Location;
                plot(app.UIAxesImage, pts1(:,1), pts1(:,2), 'go');
                
                title(app.UIAxesImage, 'Inliers in First Image');
                hold(app.UIAxesImage, 'off');
                imshow(app.modIMG2, 'Parent', app.UIAxesModify);
                hold(app.UIAxesModify, 'on');

                pts2 = app.matchedPoints2(app.inliers).Location;
                    plot(app.UIAxesModify, pts2(:,1), pts2(:,2), 'go');
                title(app.UIAxesModify, 'Inliers in Second Image');
                hold(app.UIAxesModify, 'off');
            end
        end

        % Button pushed function: EpipolarlinesIMG1Button
        function EpipolarlinesIMG1ButtonPushed(app, event)
            lines1 = epipolarLine(app.fMatrix', ...
                          app.matchedPoints2(app.inliers,:));
            borderPts1 = lineToBorderPoints(lines1, size(app.modIMG));

            hold(app.UIAxesImage, 'on');
            line(app.UIAxesImage, ...
            borderPts1(:,[1,3])', borderPts1(:,[2,4])');
            hold(app.UIAxesImage, 'off');
        end

        % Button pushed function: EpipolarlinesIMG2Button
        function EpipolarlinesIMG2ButtonPushed(app, event)
            lines2 = epipolarLine(app.fMatrix, ...
                         app.matchedPoints1(app.inliers,:));
            borderPts2 = lineToBorderPoints(lines2, size(app.modIMG2));

            hold(app.UIAxesModify, 'on');
            line(app.UIAxesModify, ...
            borderPts2(:,[1,3])', borderPts2(:,[2,4])');
            hold(app.UIAxesModify, 'off');
        end

        % Value changed function: MatchFeaturesDropDown
        function MatchFeaturesDropDownValueChanged(app, event)
            value = app.MatchFeaturesDropDown.Value;
            gray1 = im2gray(app.modIMG);
            gray2 = im2gray(app.modIMG2);
            if value =="Harris" 
                points1 = detectHarrisFeatures(gray1);
                points2 = detectHarrisFeatures(gray2);
            elseif value == "SIFT"
                points1 = detectSIFTFeatures(gray1);
                points2 = detectSIFTFeatures(gray2);
            elseif value == "SURF"
                points1 = detectSURFFeatures(gray1);
                points2 = detectSURFFeatures(gray2);

            else
                load stereoPointPairs
                app.matchedPoints1 = matchedPoints1;
                app.matchedPoints2 = matchedPoints2;
            end
            
            if value == "precomputed"
                imshow(app.modIMG, 'Parent', app.UIAxesImage);  
                hold(app.UIAxesImage, 'on');
                plot(app.UIAxesImage, app.matchedPoints1(:,1), app.matchedPoints1(:,2), 'ro');
                hold(app.UIAxesImage, 'off');
                title(app.UIAxesImage, 'Matched Points in Image 1');

                imshow(app.modIMG2, 'Parent', app.UIAxesModify);  
                hold(app.UIAxesModify, 'on');
                plot(app.UIAxesModify, app.matchedPoints2(:,1), app.matchedPoints2(:,2), 'g+');
                hold(app.UIAxesModify, 'off');
                title(app.UIAxesModify, 'Matched Points in Image 2');

                showMatchedFeatures(app.modIMG, app.modIMG2, ...
                            app.matchedPoints1, app.matchedPoints2, ...
                            'montage', 'Parent', app.UIAxesMatches);
    
                title(app.UIAxesMatches, 'Putative Matches');
            else

            [features1, validPoints1] = extractFeatures(gray1, points1);
            [features2, validPoints2] = extractFeatures(gray2, points2);

            indexPairs = matchFeatures(features1, features2, 'Unique', true);
            app.matchedPoints1 = validPoints1(indexPairs(:,1));
            app.matchedPoints2 = validPoints2(indexPairs(:,2));
            
            imshow(app.modIMG, 'Parent', app.UIAxesImage);  
            hold(app.UIAxesImage, 'on');
            plot(app.UIAxesImage, app.matchedPoints1.Location(:,1), app.matchedPoints1.Location(:,2), 'ro');
            hold(app.UIAxesImage, 'off');
            title(app.UIAxesImage, 'Matched Points in Image 1');

            imshow(app.modIMG2, 'Parent', app.UIAxesModify);  
            hold(app.UIAxesModify, 'on');
            plot(app.UIAxesModify, app.matchedPoints2.Location(:,1), app.matchedPoints2.Location(:,2), 'g+');
            hold(app.UIAxesModify, 'off');
            title(app.UIAxesModify, 'Matched Points in Image 2');

            showMatchedFeatures(app.modIMG, app.modIMG2, ...
                        app.matchedPoints1, app.matchedPoints2, ...
                        'montage', 'Parent', app.UIAxesMatches);
    
             title(app.UIAxesMatches, 'Putative Matches');
            end
        end

        % Button pushed function: LoadsecondimageButton
        function LoadsecondimageButtonPushed(app, event)
            [imageName, pathName] = uigetfile('*.*',"Pick an Image");
            imageName = strcat(pathName,imageName);
            app.IMG2 = imread(imageName);
            app.modIMG2 = app.IMG2;
            imshow(app.IMG2,'Parent',app.UIAxesModify)
            title(app.UIAxesModify,"Second Image")
        end

        % Button pushed function: RANSACButton
        function RANSACButtonPushed(app, event)
            distance = app.DistancethresholdSpinner.Value;
            confidence = app.ConfidenceSpinner.Value;
            runs = app.NumberofSamplesSpinner.Value;
            
            [tform, inlierIdx] = estimateGeometricTransform2D(...
                app.matchedPoints1, app.matchedPoints2, ...
                'affine', ...
                'MaxDistance', distance, ...         
                'Confidence', confidence, ...       
                'MaxNumTrials', runs);
                
                inlierPoints1 = app.matchedPoints1(inlierIdx);
                inlierPoints2 = app.matchedPoints2(inlierIdx);

                imshow(app.modIMG, 'Parent', app.UIAxesImage);
                hold(app.UIAxesImage, 'on');
               
                pts1 = inlierPoints1.Location;
                plot(app.UIAxesImage, pts1(:,1), pts1(:,2), 'ro');
                
                title(app.UIAxesImage, 'Inliers in First Image');
                hold(app.UIAxesImage, 'off');

                imshow(app.modIMG2, 'Parent', app.UIAxesModify);
                hold(app.UIAxesModify, 'on');
                pts2 = inlierPoints2.Location;
                plot(app.UIAxesModify, pts2(:,1), pts2(:,2), 'g+');
                
                title(app.UIAxesModify, 'Inliers in Second Image');
                hold(app.UIAxesModify, 'off');

                outlierIdx = ~inlierIdx;
                outlierPoints1 = app.matchedPoints1(outlierIdx);
                outlierPoints2 = app.matchedPoints2(outlierIdx);
                

                hold(app.UIAxesransac, 'on');
                showMatchedFeatures(app.modIMG, app.modIMG2, ...
                    outlierPoints1, outlierPoints2, ...
                    'montage', 'Parent', app.UIAxesransac, ...
                    'PlotOptions', {'go','go','r--'});
                hold(app.UIAxesransac, 'off');

                title(app.UIAxesransac, 'rejected Outliers ');
        end

        % Value changed function: MatchFeaturesDropDown_2
        function MatchFeaturesDropDown_2ValueChanged(app, event)
            value = app.MatchFeaturesDropDown_2.Value;
            gray1 = im2gray(app.modIMG);
            gray2 = im2gray(app.modIMG2);
            if value =="Harris" 
                points1 = detectHarrisFeatures(gray1);
                points2 = detectHarrisFeatures(gray2);
            elseif value == "SIFT"
                points1 = detectSIFTFeatures(gray1);
                points2 = detectSIFTFeatures(gray2);
            else
                points1 = detectSURFFeatures(gray1);
                points2 = detectSURFFeatures(gray2);
            end
            
            [features1, validPoints1] = extractFeatures(gray1, points1);
            [features2, validPoints2] = extractFeatures(gray2, points2);

            indexPairs = matchFeatures(features1, features2, 'Unique', true);
            app.matchedPoints1 = validPoints1(indexPairs(:,1));
            app.matchedPoints2 = validPoints2(indexPairs(:,2));
            
            imshow(app.modIMG, 'Parent', app.UIAxesImage);  
            hold(app.UIAxesImage, 'on');
            plot(app.UIAxesImage, app.matchedPoints1.Location(:,1), app.matchedPoints1.Location(:,2), 'ro');
            hold(app.UIAxesImage, 'off');
            title(app.UIAxesImage, 'Matched Points in Image 1');

            imshow(app.modIMG2, 'Parent', app.UIAxesModify);  
            hold(app.UIAxesModify, 'on');
            plot(app.UIAxesModify, app.matchedPoints2.Location(:,1), app.matchedPoints2.Location(:,2), 'g+');
            hold(app.UIAxesModify, 'off');
            title(app.UIAxesModify, 'Matched Points in Image 2');

            showMatchedFeatures(app.modIMG, app.modIMG2, ...
                        app.matchedPoints1, app.matchedPoints2, ...
                        'montage', 'Parent', app.UIAxesransac);
    
             title(app.UIAxesransac, 'Putative Matches');
        end

        % Button pushed function: LinedetectionButton
        function LinedetectionButtonPushed(app, event)
            dthresh = app.DistancethresholdSpinner.Value;
            iterations = app.samplenumberSpinner.Value;
      
            gray = im2gray(app.modIMG);
            edges = edge(gray, 'canny');
            [y, x] = find(edges); 
            points = [x, y];

            [bestLine, inliers] = ransacLine(app, points, dthresh, iterations);
            
            cla(app.UIAxesModify);
            imshow(app.modIMG, 'Parent', app.UIAxesModify);
            axis(app.UIAxesModify, 'image');
            hold(app.UIAxesModify,"on");
            plot(app.UIAxesModify,points(inliers,1), points(inliers,2), 'g.'); 
            if abs(bestLine(2)) > 1e-6
                x_vals = [1, size(gray, 2)];
                y_vals = -(bestLine(1)*x_vals + bestLine(3)) / bestLine(2);
                plot(app.UIAxesModify,x_vals, y_vals, 'g-', 'LineWidth', 2);
            else
            x_val = -bestLine(3)/bestLine(1);
            y_vals = [1, size(gray, 1)];
            plot(app.UIAxesModify,[x_val, x_val], y_vals, 'g-', 'LineWidth', 2);
            end
            hold(app.UIAxesModify,"off");
        end

        % Button pushed function: CircledetectionButton
        function CircledetectionButtonPushed(app, event)
            dthresh = app.DistancethresholdSpinner.Value;
            iterations = app.samplenumberSpinner.Value;

            gray = im2gray(app.modIMG);
            edges = edge(gray, 'canny');

            [centersFound, radiiFound, metrics] = imfindcircles(gray, [20 100], ...
                'Sensitivity', 0.95);

            mask = false(size(gray));
            for i = 1:min(3, size(centersFound,1))  
                mask = mask | createCircularMask(app,size(gray), centersFound(i,:), radiiFound(i) + 5);
            end

            [y, x] = find(edges & mask);
            points = [x, y];

            [center, radius, inliers] = ransacCircle(app,points, dthresh, iterations);

            
            cla(app.UIAxesModify);
            imshow(app.modIMG, 'Parent', app.UIAxesModify);
            axis(app.UIAxesModify, 'image');
            hold(app.UIAxesModify,"on");
            plot(app.UIAxesModify,points(inliers,1), points(inliers,2), 'g.'); 
            viscircles(app.UIAxesModify,center, radius, 'EdgeColor', 'b');
            title(app.UIAxesModify,'RANSAC Circle Detection ');
            hold(app.UIAxesModify,"off");
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 727 598];
            app.UIFigure.Name = 'MATLAB App';

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {169.67, '1x', '1.13x'};
            app.GridLayout.RowHeight = {221, '1x'};
            app.GridLayout.ColumnSpacing = 0;
            app.GridLayout.RowSpacing = 9;
            app.GridLayout.Padding = [0 9 0 9];

            % Create UIAxesModify
            app.UIAxesModify = uiaxes(app.GridLayout);
            title(app.UIAxesModify, 'Modified')
            zlabel(app.UIAxesModify, 'Z')
            app.UIAxesModify.XTick = [];
            app.UIAxesModify.YTick = [];
            app.UIAxesModify.Layout.Row = 1;
            app.UIAxesModify.Layout.Column = 3;

            % Create UIAxesImage
            app.UIAxesImage = uiaxes(app.GridLayout);
            title(app.UIAxesImage, 'Original')
            zlabel(app.UIAxesImage, 'Z')
            app.UIAxesImage.XTick = [];
            app.UIAxesImage.XTickLabel = '';
            app.UIAxesImage.YTick = [];
            app.UIAxesImage.Layout.Row = 1;
            app.UIAxesImage.Layout.Column = 2;

            % Create BasIcOperationsPanel
            app.BasIcOperationsPanel = uipanel(app.GridLayout);
            app.BasIcOperationsPanel.Title = 'BasIc Operations ';
            app.BasIcOperationsPanel.Layout.Row = 1;
            app.BasIcOperationsPanel.Layout.Column = 1;

            % Create loadimageButton
            app.loadimageButton = uibutton(app.BasIcOperationsPanel, 'push');
            app.loadimageButton.ButtonPushedFcn = createCallbackFcn(app, @loadimageButtonPushed, true);
            app.loadimageButton.Position = [39 167 100 22];
            app.loadimageButton.Text = 'load image';

            % Create GreyscaleButton
            app.GreyscaleButton = uibutton(app.BasIcOperationsPanel, 'push');
            app.GreyscaleButton.ButtonPushedFcn = createCallbackFcn(app, @GreyscaleButtonPushed, true);
            app.GreyscaleButton.Enable = 'off';
            app.GreyscaleButton.Position = [39 68 100 22];
            app.GreyscaleButton.Text = 'Greyscale';

            % Create BlackandWhiteButton
            app.BlackandWhiteButton = uibutton(app.BasIcOperationsPanel, 'push');
            app.BlackandWhiteButton.ButtonPushedFcn = createCallbackFcn(app, @BlackandWhiteButtonPushed, true);
            app.BlackandWhiteButton.Enable = 'off';
            app.BlackandWhiteButton.Position = [39 26 101 22];
            app.BlackandWhiteButton.Text = 'Black and White';

            % Create RESETButton
            app.RESETButton = uibutton(app.BasIcOperationsPanel, 'push');
            app.RESETButton.ButtonPushedFcn = createCallbackFcn(app, @RESETButtonPushed, true);
            app.RESETButton.Enable = 'off';
            app.RESETButton.Position = [39 121 100 22];
            app.RESETButton.Text = 'RESET';

            % Create TabGroup
            app.TabGroup = uitabgroup(app.GridLayout);
            app.TabGroup.Visible = 'off';
            app.TabGroup.Layout.Row = 2;
            app.TabGroup.Layout.Column = [1 3];

            % Create imageenhancementTab
            app.imageenhancementTab = uitab(app.TabGroup);
            app.imageenhancementTab.Title = 'image enhancement ';

            % Create UIAxes_HistO
            app.UIAxes_HistO = uiaxes(app.imageenhancementTab);
            title(app.UIAxes_HistO, 'Title')
            xlabel(app.UIAxes_HistO, 'X')
            ylabel(app.UIAxes_HistO, 'Y')
            zlabel(app.UIAxes_HistO, 'Z')
            app.UIAxes_HistO.Position = [153 37 266 185];

            % Create UIAxes_HistN
            app.UIAxes_HistN = uiaxes(app.imageenhancementTab);
            title(app.UIAxes_HistN, 'Title')
            xlabel(app.UIAxes_HistN, 'X')
            ylabel(app.UIAxes_HistN, 'Y')
            zlabel(app.UIAxes_HistN, 'Z')
            app.UIAxes_HistN.Position = [438 37 264 185];

            % Create brightenSliderLabel
            app.brightenSliderLabel = uilabel(app.imageenhancementTab);
            app.brightenSliderLabel.HorizontalAlignment = 'right';
            app.brightenSliderLabel.Position = [170 262 48 22];
            app.brightenSliderLabel.Text = 'brighten';

            % Create brightenSlider
            app.brightenSlider = uislider(app.imageenhancementTab);
            app.brightenSlider.Limits = [1 30];
            app.brightenSlider.ValueChangedFcn = createCallbackFcn(app, @brightenSliderValueChanged, true);
            app.brightenSlider.Position = [239 271 150 3];
            app.brightenSlider.Value = 1;

            % Create histogramButton
            app.histogramButton = uibutton(app.imageenhancementTab, 'push');
            app.histogramButton.ButtonPushedFcn = createCallbackFcn(app, @histogramButtonPushed, true);
            app.histogramButton.Position = [24 183 100 22];
            app.histogramButton.Text = 'histogram';

            % Create darkenSliderLabel
            app.darkenSliderLabel = uilabel(app.imageenhancementTab);
            app.darkenSliderLabel.HorizontalAlignment = 'right';
            app.darkenSliderLabel.Position = [445 262 42 22];
            app.darkenSliderLabel.Text = 'darken';

            % Create darkenSlider
            app.darkenSlider = uislider(app.imageenhancementTab);
            app.darkenSlider.Limits = [0 1];
            app.darkenSlider.ValueChangedFcn = createCallbackFcn(app, @darkenSliderValueChanged, true);
            app.darkenSlider.Position = [508 271 150 3];

            % Create SpatialdomainfilteringTab
            app.SpatialdomainfilteringTab = uitab(app.TabGroup);
            app.SpatialdomainfilteringTab.Title = 'Spatial domain filtering ';

            % Create SmoothPanel
            app.SmoothPanel = uipanel(app.SpatialdomainfilteringTab);
            app.SmoothPanel.Title = 'Smooth ';
            app.SmoothPanel.Position = [24 137 260 173];

            % Create BoxFilterCheckBox
            app.BoxFilterCheckBox = uicheckbox(app.SmoothPanel);
            app.BoxFilterCheckBox.ValueChangedFcn = createCallbackFcn(app, @BoxFilterCheckBoxValueChanged, true);
            app.BoxFilterCheckBox.Enable = 'off';
            app.BoxFilterCheckBox.Text = 'Box Filter';
            app.BoxFilterCheckBox.Position = [26 114 72 22];

            % Create WeightedAverageCheckBox
            app.WeightedAverageCheckBox = uicheckbox(app.SmoothPanel);
            app.WeightedAverageCheckBox.ValueChangedFcn = createCallbackFcn(app, @WeightedAverageCheckBoxValueChanged, true);
            app.WeightedAverageCheckBox.Enable = 'off';
            app.WeightedAverageCheckBox.Text = 'Weighted Average';
            app.WeightedAverageCheckBox.Position = [59 70 119 22];

            % Create MedianCheckBox
            app.MedianCheckBox = uicheckbox(app.SmoothPanel);
            app.MedianCheckBox.ValueChangedFcn = createCallbackFcn(app, @MedianCheckBoxValueChanged, true);
            app.MedianCheckBox.Enable = 'off';
            app.MedianCheckBox.Text = 'Median ';
            app.MedianCheckBox.Position = [138 114 64 22];

            % Create filterSizeSpinnerLabel
            app.filterSizeSpinnerLabel = uilabel(app.SmoothPanel);
            app.filterSizeSpinnerLabel.HorizontalAlignment = 'right';
            app.filterSizeSpinnerLabel.Position = [38 22 58 22];
            app.filterSizeSpinnerLabel.Text = 'filter Size ';

            % Create filterSizeSpinner
            app.filterSizeSpinner = uispinner(app.SmoothPanel);
            app.filterSizeSpinner.Step = 2;
            app.filterSizeSpinner.Limits = [3 9];
            app.filterSizeSpinner.ValueChangedFcn = createCallbackFcn(app, @filterSizeSpinnerValueChanged, true);
            app.filterSizeSpinner.Position = [111 22 100 22];
            app.filterSizeSpinner.Value = 3;

            % Create SharpenPanel
            app.SharpenPanel = uipanel(app.SpatialdomainfilteringTab);
            app.SharpenPanel.Title = 'Sharpen';
            app.SharpenPanel.Position = [321 16 273 294];

            % Create LaplacianPanel
            app.LaplacianPanel = uipanel(app.SharpenPanel);
            app.LaplacianPanel.Title = 'Laplacian';
            app.LaplacianPanel.Position = [16 164 241 103];

            % Create firstderivativeButton
            app.firstderivativeButton = uibutton(app.LaplacianPanel, 'push');
            app.firstderivativeButton.ButtonPushedFcn = createCallbackFcn(app, @firstderivativeButtonPushed, true);
            app.firstderivativeButton.Position = [8 52 100 22];
            app.firstderivativeButton.Text = 'first derivative';

            % Create secondderivativeButton
            app.secondderivativeButton = uibutton(app.LaplacianPanel, 'push');
            app.secondderivativeButton.ButtonPushedFcn = createCallbackFcn(app, @secondderivativeButtonPushed, true);
            app.secondderivativeButton.Position = [124 52 108 22];
            app.secondderivativeButton.Text = 'second derivative';

            % Create boostingAlphaEditFieldLabel
            app.boostingAlphaEditFieldLabel = uilabel(app.LaplacianPanel);
            app.boostingAlphaEditFieldLabel.HorizontalAlignment = 'right';
            app.boostingAlphaEditFieldLabel.Position = [18 7 87 22];
            app.boostingAlphaEditFieldLabel.Text = 'boosting Alpha ';

            % Create boostingAlphaEditField
            app.boostingAlphaEditField = uieditfield(app.LaplacianPanel, 'numeric');
            app.boostingAlphaEditField.Limits = [0 Inf];
            app.boostingAlphaEditField.ValueChangedFcn = createCallbackFcn(app, @boostingAlphaEditFieldValueChanged, true);
            app.boostingAlphaEditField.Position = [120 7 100 22];

            % Create SobelPanel
            app.SobelPanel = uipanel(app.SharpenPanel);
            app.SobelPanel.Title = 'Sobel';
            app.SobelPanel.Position = [16 46 241 104];

            % Create horizontalButton
            app.horizontalButton = uibutton(app.SobelPanel, 'push');
            app.horizontalButton.ButtonPushedFcn = createCallbackFcn(app, @horizontalButtonPushed, true);
            app.horizontalButton.Position = [11 48 100 22];
            app.horizontalButton.Text = 'horizontal';

            % Create verticalButton
            app.verticalButton = uibutton(app.SobelPanel, 'push');
            app.verticalButton.ButtonPushedFcn = createCallbackFcn(app, @verticalButtonPushed, true);
            app.verticalButton.Position = [128 49 100 22];
            app.verticalButton.Text = 'vertical';

            % Create boostingAlphaEditField_2Label
            app.boostingAlphaEditField_2Label = uilabel(app.SobelPanel);
            app.boostingAlphaEditField_2Label.HorizontalAlignment = 'right';
            app.boostingAlphaEditField_2Label.Position = [25 3 87 22];
            app.boostingAlphaEditField_2Label.Text = 'boosting Alpha ';

            % Create boostingAlphaEditField_2
            app.boostingAlphaEditField_2 = uieditfield(app.SobelPanel, 'numeric');
            app.boostingAlphaEditField_2.Limits = [0 Inf];
            app.boostingAlphaEditField_2.ValueChangedFcn = createCallbackFcn(app, @boostingAlphaEditField_2ValueChanged, true);
            app.boostingAlphaEditField_2.Position = [127 3 100 22];

            % Create PrewittButton
            app.PrewittButton = uibutton(app.SharpenPanel, 'push');
            app.PrewittButton.ButtonPushedFcn = createCallbackFcn(app, @PrewittButtonPushed, true);
            app.PrewittButton.Position = [16 11 100 22];
            app.PrewittButton.Text = 'Prewitt';

            % Create FrequnecydomainfilteringTab
            app.FrequnecydomainfilteringTab = uitab(app.TabGroup);
            app.FrequnecydomainfilteringTab.Title = 'Frequnecy domain filtering ';

            % Create SmoothPanel_2
            app.SmoothPanel_2 = uipanel(app.FrequnecydomainfilteringTab);
            app.SmoothPanel_2.Title = 'Smooth';
            app.SmoothPanel_2.Position = [62 99 260 196];

            % Create IdealButterworthfilteringLPFCheckBox
            app.IdealButterworthfilteringLPFCheckBox = uicheckbox(app.SmoothPanel_2);
            app.IdealButterworthfilteringLPFCheckBox.ValueChangedFcn = createCallbackFcn(app, @IdealButterworthfilteringLPFCheckBoxValueChanged, true);
            app.IdealButterworthfilteringLPFCheckBox.Text = 'Ideal Butterworth filtering  LPF';
            app.IdealButterworthfilteringLPFCheckBox.Position = [18 141 183 22];

            % Create GaussianlowpassfilteringCheckBox
            app.GaussianlowpassfilteringCheckBox = uicheckbox(app.SmoothPanel_2);
            app.GaussianlowpassfilteringCheckBox.ValueChangedFcn = createCallbackFcn(app, @GaussianlowpassfilteringCheckBoxValueChanged, true);
            app.GaussianlowpassfilteringCheckBox.Text = 'Gaussian low pass filtering ';
            app.GaussianlowpassfilteringCheckBox.Position = [26 65 167 22];

            % Create OrderSpinnerLabel
            app.OrderSpinnerLabel = uilabel(app.SmoothPanel_2);
            app.OrderSpinnerLabel.HorizontalAlignment = 'right';
            app.OrderSpinnerLabel.Position = [1 109 36 22];
            app.OrderSpinnerLabel.Text = 'Order';

            % Create OrderSpinner
            app.OrderSpinner = uispinner(app.SmoothPanel_2);
            app.OrderSpinner.Limits = [1 Inf];
            app.OrderSpinner.ValueChangedFcn = createCallbackFcn(app, @OrderSpinnerValueChanged, true);
            app.OrderSpinner.Position = [52 109 49 22];
            app.OrderSpinner.Value = 1;

            % Create cutofffrequencySpinnerLabel
            app.cutofffrequencySpinnerLabel = uilabel(app.SmoothPanel_2);
            app.cutofffrequencySpinnerLabel.HorizontalAlignment = 'right';
            app.cutofffrequencySpinnerLabel.Position = [107 108 90 22];
            app.cutofffrequencySpinnerLabel.Text = 'cutoff frequency';

            % Create cutofffrequencySpinner
            app.cutofffrequencySpinner = uispinner(app.SmoothPanel_2);
            app.cutofffrequencySpinner.Limits = [10 217];
            app.cutofffrequencySpinner.ValueChangedFcn = createCallbackFcn(app, @cutofffrequencySpinnerValueChanged, true);
            app.cutofffrequencySpinner.Position = [211 108 49 22];
            app.cutofffrequencySpinner.Value = 10;

            % Create cutofffrequencySpinner_3Label
            app.cutofffrequencySpinner_3Label = uilabel(app.SmoothPanel_2);
            app.cutofffrequencySpinner_3Label.HorizontalAlignment = 'right';
            app.cutofffrequencySpinner_3Label.Position = [36 33 90 22];
            app.cutofffrequencySpinner_3Label.Text = 'cutoff frequency';

            % Create cutofffrequencySpinner_3
            app.cutofffrequencySpinner_3 = uispinner(app.SmoothPanel_2);
            app.cutofffrequencySpinner_3.Limits = [10 100];
            app.cutofffrequencySpinner_3.ValueChangedFcn = createCallbackFcn(app, @cutofffrequencySpinner_3ValueChanged, true);
            app.cutofffrequencySpinner_3.Position = [141 33 70 22];
            app.cutofffrequencySpinner_3.Value = 30;

            % Create SharpenPanel_2
            app.SharpenPanel_2 = uipanel(app.FrequnecydomainfilteringTab);
            app.SharpenPanel_2.Title = 'Sharpen ';
            app.SharpenPanel_2.Position = [361 99 260 196];

            % Create IdealButterworthfilteringHPFCheckBox
            app.IdealButterworthfilteringHPFCheckBox = uicheckbox(app.SharpenPanel_2);
            app.IdealButterworthfilteringHPFCheckBox.ValueChangedFcn = createCallbackFcn(app, @IdealButterworthfilteringHPFCheckBoxValueChanged, true);
            app.IdealButterworthfilteringHPFCheckBox.Text = 'Ideal Butterworth filtering  HPF';
            app.IdealButterworthfilteringHPFCheckBox.Position = [31 140 185 22];

            % Create GaussianHighpassfilteringCheckBox
            app.GaussianHighpassfilteringCheckBox = uicheckbox(app.SharpenPanel_2);
            app.GaussianHighpassfilteringCheckBox.ValueChangedFcn = createCallbackFcn(app, @GaussianHighpassfilteringCheckBoxValueChanged, true);
            app.GaussianHighpassfilteringCheckBox.Text = 'Gaussian High pass filtering ';
            app.GaussianHighpassfilteringCheckBox.Position = [31 66 174 22];

            % Create orderSpinnerLabel
            app.orderSpinnerLabel = uilabel(app.SharpenPanel_2);
            app.orderSpinnerLabel.HorizontalAlignment = 'right';
            app.orderSpinnerLabel.Position = [1 108 33 22];
            app.orderSpinnerLabel.Text = 'order';

            % Create orderSpinner
            app.orderSpinner = uispinner(app.SharpenPanel_2);
            app.orderSpinner.Limits = [1 Inf];
            app.orderSpinner.ValueChangedFcn = createCallbackFcn(app, @orderSpinnerValueChanged, true);
            app.orderSpinner.Position = [49 108 49 22];
            app.orderSpinner.Value = 1;

            % Create cutofffrequencySpinner_2Label
            app.cutofffrequencySpinner_2Label = uilabel(app.SharpenPanel_2);
            app.cutofffrequencySpinner_2Label.HorizontalAlignment = 'right';
            app.cutofffrequencySpinner_2Label.Position = [103 109 90 22];
            app.cutofffrequencySpinner_2Label.Text = 'cutoff frequency';

            % Create cutofffrequencySpinner_2
            app.cutofffrequencySpinner_2 = uispinner(app.SharpenPanel_2);
            app.cutofffrequencySpinner_2.Limits = [1 217];
            app.cutofffrequencySpinner_2.ValueChangedFcn = createCallbackFcn(app, @cutofffrequencySpinner_2ValueChanged, true);
            app.cutofffrequencySpinner_2.Position = [208 109 52 22];
            app.cutofffrequencySpinner_2.Value = 1;

            % Create cutofffrequencySpinner_4Label
            app.cutofffrequencySpinner_4Label = uilabel(app.SharpenPanel_2);
            app.cutofffrequencySpinner_4Label.HorizontalAlignment = 'right';
            app.cutofffrequencySpinner_4Label.Position = [39 33 90 22];
            app.cutofffrequencySpinner_4Label.Text = 'cutoff frequency';

            % Create cutofffrequencySpinner_4
            app.cutofffrequencySpinner_4 = uispinner(app.SharpenPanel_2);
            app.cutofffrequencySpinner_4.Limits = [10 100];
            app.cutofffrequencySpinner_4.ValueChangedFcn = createCallbackFcn(app, @cutofffrequencySpinner_4ValueChanged, true);
            app.cutofffrequencySpinner_4.Position = [144 33 72 22];
            app.cutofffrequencySpinner_4.Value = 30;

            % Create ColorSpaceTab
            app.ColorSpaceTab = uitab(app.TabGroup);
            app.ColorSpaceTab.Title = 'Color Space';

            % Create UIAxes
            app.UIAxes = uiaxes(app.ColorSpaceTab);
            title(app.UIAxes, 'Title')
            zlabel(app.UIAxes, 'Z')
            app.UIAxes.XTick = [];
            app.UIAxes.YTick = [];
            app.UIAxes.Position = [103 137 193 185];

            % Create UIAxes2
            app.UIAxes2 = uiaxes(app.ColorSpaceTab);
            title(app.UIAxes2, 'Title')
            zlabel(app.UIAxes2, 'Z')
            app.UIAxes2.XTick = [];
            app.UIAxes2.YTick = [];
            app.UIAxes2.Position = [306 137 198 185];

            % Create UIAxes3
            app.UIAxes3 = uiaxes(app.ColorSpaceTab);
            title(app.UIAxes3, 'Title')
            zlabel(app.UIAxes3, 'Z')
            app.UIAxes3.XTick = [];
            app.UIAxes3.YTick = [];
            app.UIAxes3.Position = [508 137 201 185];

            % Create RGBCheckBox
            app.RGBCheckBox = uicheckbox(app.ColorSpaceTab);
            app.RGBCheckBox.ValueChangedFcn = createCallbackFcn(app, @RGBCheckBoxValueChanged, true);
            app.RGBCheckBox.Text = 'RGB';
            app.RGBCheckBox.Position = [33 279 47 22];

            % Create HSICheckBox
            app.HSICheckBox = uicheckbox(app.ColorSpaceTab);
            app.HSICheckBox.ValueChangedFcn = createCallbackFcn(app, @HSICheckBoxValueChanged, true);
            app.HSICheckBox.Text = 'HSI';
            app.HSICheckBox.Position = [33 230 41 22];

            % Create LabCheckBox
            app.LabCheckBox = uicheckbox(app.ColorSpaceTab);
            app.LabCheckBox.ValueChangedFcn = createCallbackFcn(app, @LabCheckBoxValueChanged, true);
            app.LabCheckBox.Text = 'L*a*b*';
            app.LabCheckBox.Position = [33 177 55 22];

            % Create YCbCrCheckBox
            app.YCbCrCheckBox = uicheckbox(app.ColorSpaceTab);
            app.YCbCrCheckBox.ValueChangedFcn = createCallbackFcn(app, @YCbCrCheckBoxValueChanged, true);
            app.YCbCrCheckBox.Text = 'YCbCr';
            app.YCbCrCheckBox.Position = [33 121 57 22];

            % Create PyramidsFiltersTab
            app.PyramidsFiltersTab = uitab(app.TabGroup);
            app.PyramidsFiltersTab.Title = 'Pyramids & Filters';

            % Create UIAxesTemplate
            app.UIAxesTemplate = uiaxes(app.PyramidsFiltersTab);
            title(app.UIAxesTemplate, 'Title')
            app.UIAxesTemplate.XTick = [];
            app.UIAxesTemplate.YTick = [];
            app.UIAxesTemplate.Position = [272 68 199 185];

            % Create PyramidsPanel
            app.PyramidsPanel = uipanel(app.PyramidsFiltersTab);
            app.PyramidsPanel.Title = 'Pyramids ';
            app.PyramidsPanel.Position = [12 204 250 116];

            % Create GuassianPyramidCheckBox
            app.GuassianPyramidCheckBox = uicheckbox(app.PyramidsPanel);
            app.GuassianPyramidCheckBox.ValueChangedFcn = createCallbackFcn(app, @GuassianPyramidCheckBoxValueChanged, true);
            app.GuassianPyramidCheckBox.Enable = 'off';
            app.GuassianPyramidCheckBox.Text = 'Guassian Pyramid';
            app.GuassianPyramidCheckBox.Position = [13 66 119 22];

            % Create ReductiontimesSpinnerLabel
            app.ReductiontimesSpinnerLabel = uilabel(app.PyramidsPanel);
            app.ReductiontimesSpinnerLabel.HorizontalAlignment = 'right';
            app.ReductiontimesSpinnerLabel.Position = [17 36 91 22];
            app.ReductiontimesSpinnerLabel.Text = 'Reduction times';

            % Create ReductiontimesSpinner
            app.ReductiontimesSpinner = uispinner(app.PyramidsPanel);
            app.ReductiontimesSpinner.Limits = [1 Inf];
            app.ReductiontimesSpinner.ValueChangedFcn = createCallbackFcn(app, @ReductiontimesSpinnerValueChanged, true);
            app.ReductiontimesSpinner.Position = [122 36 62 22];
            app.ReductiontimesSpinner.Value = 1;

            % Create LaplacianPyramidCheckBox
            app.LaplacianPyramidCheckBox = uicheckbox(app.PyramidsPanel);
            app.LaplacianPyramidCheckBox.ValueChangedFcn = createCallbackFcn(app, @LaplacianPyramidCheckBoxValueChanged, true);
            app.LaplacianPyramidCheckBox.Enable = 'off';
            app.LaplacianPyramidCheckBox.Text = 'Laplacian Pyramid';
            app.LaplacianPyramidCheckBox.Position = [3 7 120 22];

            % Create levelSpinnerLabel
            app.levelSpinnerLabel = uilabel(app.PyramidsPanel);
            app.levelSpinnerLabel.HorizontalAlignment = 'right';
            app.levelSpinnerLabel.Enable = 'off';
            app.levelSpinnerLabel.Position = [131 7 30 22];
            app.levelSpinnerLabel.Text = 'level';

            % Create levelSpinner
            app.levelSpinner = uispinner(app.PyramidsPanel);
            app.levelSpinner.Limits = [1 Inf];
            app.levelSpinner.ValueChangedFcn = createCallbackFcn(app, @levelSpinnerValueChanged, true);
            app.levelSpinner.Enable = 'off';
            app.levelSpinner.Position = [176 7 54 22];
            app.levelSpinner.Value = 1;

            % Create LevelSpinner
            app.LevelSpinner = uispinner(app.PyramidsPanel);
            app.LevelSpinner.Limits = [1 Inf];
            app.LevelSpinner.ValueChangedFcn = createCallbackFcn(app, @LevelSpinnerValueChanged, true);
            app.LevelSpinner.Enable = 'off';
            app.LevelSpinner.Position = [190 66 47 22];
            app.LevelSpinner.Value = 1;

            % Create LevelSpinnerLabel
            app.LevelSpinnerLabel = uilabel(app.PyramidsPanel);
            app.LevelSpinnerLabel.HorizontalAlignment = 'right';
            app.LevelSpinnerLabel.Enable = 'off';
            app.LevelSpinnerLabel.Position = [141 66 34 22];
            app.LevelSpinnerLabel.Text = 'Level';

            % Create TemplateMatcingPanel
            app.TemplateMatcingPanel = uipanel(app.PyramidsFiltersTab);
            app.TemplateMatcingPanel.Title = 'Template Matcing';
            app.TemplateMatcingPanel.Position = [495 32 207 279];

            % Create LoadTemplateButton
            app.LoadTemplateButton = uibutton(app.TemplateMatcingPanel, 'push');
            app.LoadTemplateButton.ButtonPushedFcn = createCallbackFcn(app, @LoadTemplateButtonPushed, true);
            app.LoadTemplateButton.Position = [22 223 100 22];
            app.LoadTemplateButton.Text = 'Load Template';

            % Create SelecttemplatefromimageButton
            app.SelecttemplatefromimageButton = uibutton(app.TemplateMatcingPanel, 'push');
            app.SelecttemplatefromimageButton.ButtonPushedFcn = createCallbackFcn(app, @SelecttemplatefromimageButtonPushed, true);
            app.SelecttemplatefromimageButton.Position = [23 191 164 22];
            app.SelecttemplatefromimageButton.Text = 'Select template from image ';

            % Create ZeromeanCorrelationCheckBox
            app.ZeromeanCorrelationCheckBox = uicheckbox(app.TemplateMatcingPanel);
            app.ZeromeanCorrelationCheckBox.ValueChangedFcn = createCallbackFcn(app, @ZeromeanCorrelationCheckBoxValueChanged, true);
            app.ZeromeanCorrelationCheckBox.Text = 'Zero mean Correlation';
            app.ZeromeanCorrelationCheckBox.Position = [22 154 141 22];

            % Create SumSquareDifferenceCheckBox
            app.SumSquareDifferenceCheckBox = uicheckbox(app.TemplateMatcingPanel);
            app.SumSquareDifferenceCheckBox.ValueChangedFcn = createCallbackFcn(app, @SumSquareDifferenceCheckBoxValueChanged, true);
            app.SumSquareDifferenceCheckBox.Text = 'Sum Square Difference';
            app.SumSquareDifferenceCheckBox.Position = [22 120 146 22];

            % Create NormalizedcrosscorrelationCheckBox
            app.NormalizedcrosscorrelationCheckBox = uicheckbox(app.TemplateMatcingPanel);
            app.NormalizedcrosscorrelationCheckBox.ValueChangedFcn = createCallbackFcn(app, @NormalizedcrosscorrelationCheckBoxValueChanged, true);
            app.NormalizedcrosscorrelationCheckBox.Text = 'Normalized cross correlation';
            app.NormalizedcrosscorrelationCheckBox.Position = [22 88 173 22];

            % Create GuassianPyramidsCheckBox
            app.GuassianPyramidsCheckBox = uicheckbox(app.TemplateMatcingPanel);
            app.GuassianPyramidsCheckBox.ValueChangedFcn = createCallbackFcn(app, @GuassianPyramidsCheckBoxValueChanged, true);
            app.GuassianPyramidsCheckBox.Text = 'Guassian Pyramids';
            app.GuassianPyramidsCheckBox.Position = [22 54 125 22];

            % Create FilterBanksPanel
            app.FilterBanksPanel = uipanel(app.PyramidsFiltersTab);
            app.FilterBanksPanel.Title = 'Filter Banks';
            app.FilterBanksPanel.Position = [12 1 250 183];

            % Create FilterTypeDropDownLabel
            app.FilterTypeDropDownLabel = uilabel(app.FilterBanksPanel);
            app.FilterTypeDropDownLabel.HorizontalAlignment = 'right';
            app.FilterTypeDropDownLabel.Position = [28 119 61 22];
            app.FilterTypeDropDownLabel.Text = 'Filter Type';

            % Create FilterTypeDropDown
            app.FilterTypeDropDown = uidropdown(app.FilterBanksPanel);
            app.FilterTypeDropDown.Items = {'Bar ', 'Edge', 'Gaussian', 'LoG (sigma)', 'LoG (3 sigma)'};
            app.FilterTypeDropDown.ValueChangedFcn = createCallbackFcn(app, @FilterTypeDropDownValueChanged, true);
            app.FilterTypeDropDown.Position = [103 119 100 22];
            app.FilterTypeDropDown.Value = 'Bar ';

            % Create EdgeFiltersLabel
            app.EdgeFiltersLabel = uilabel(app.FilterBanksPanel);
            app.EdgeFiltersLabel.HorizontalAlignment = 'right';
            app.EdgeFiltersLabel.Position = [26 79 64 22];
            app.EdgeFiltersLabel.Text = 'Orientation';

            % Create OrientationDropDown
            app.OrientationDropDown = uidropdown(app.FilterBanksPanel);
            app.OrientationDropDown.Items = {'0 degrees', '30 degrees', '60 degrees', '90  degrees', '120 degrees', '150 degrees'};
            app.OrientationDropDown.Position = [105 79 100 22];
            app.OrientationDropDown.Value = '0 degrees';

            % Create ScaleDropDownLabel
            app.ScaleDropDownLabel = uilabel(app.FilterBanksPanel);
            app.ScaleDropDownLabel.HorizontalAlignment = 'right';
            app.ScaleDropDownLabel.Position = [50 44 35 22];
            app.ScaleDropDownLabel.Text = 'Scale';

            % Create ScaleDropDown
            app.ScaleDropDown = uidropdown(app.FilterBanksPanel);
            app.ScaleDropDown.Items = {'1', '2', '3', '4'};
            app.ScaleDropDown.Position = [100 44 100 22];
            app.ScaleDropDown.Value = '1';

            % Create ApplyselectedfilterButton
            app.ApplyselectedfilterButton = uibutton(app.FilterBanksPanel, 'push');
            app.ApplyselectedfilterButton.ButtonPushedFcn = createCallbackFcn(app, @ApplyselectedfilterButtonPushed, true);
            app.ApplyselectedfilterButton.Position = [71 5 122 22];
            app.ApplyselectedfilterButton.Text = 'Apply selected filter ';

            % Create edgeandcornerTab
            app.edgeandcornerTab = uitab(app.TabGroup);
            app.edgeandcornerTab.Title = 'edge and corner';

            % Create EdgedetectionPanel
            app.EdgedetectionPanel = uipanel(app.edgeandcornerTab);
            app.EdgedetectionPanel.Title = 'Edge detection';
            app.EdgedetectionPanel.Position = [21 49 275 255];

            % Create CannyEdgedetectionButton
            app.CannyEdgedetectionButton = uibutton(app.EdgedetectionPanel, 'push');
            app.CannyEdgedetectionButton.ButtonPushedFcn = createCallbackFcn(app, @CannyEdgedetectionButtonPushed, true);
            app.CannyEdgedetectionButton.Position = [68 200 132 22];
            app.CannyEdgedetectionButton.Text = 'Canny Edge detection';

            % Create thresholdSliderLabel
            app.thresholdSliderLabel = uilabel(app.EdgedetectionPanel);
            app.thresholdSliderLabel.HorizontalAlignment = 'right';
            app.thresholdSliderLabel.Position = [19 79 54 22];
            app.thresholdSliderLabel.Text = 'threshold';

            % Create thresholdSlider
            app.thresholdSlider = uislider(app.EdgedetectionPanel, 'range');
            app.thresholdSlider.Limits = [0 1];
            app.thresholdSlider.ValueChangedFcn = createCallbackFcn(app, @thresholdSliderValueChanged, true);
            app.thresholdSlider.Position = [95 88 150 3];
            app.thresholdSlider.Value = [0 1];

            % Create SigmaSliderLabel
            app.SigmaSliderLabel = uilabel(app.EdgedetectionPanel);
            app.SigmaSliderLabel.HorizontalAlignment = 'right';
            app.SigmaSliderLabel.Position = [29 154 42 22];
            app.SigmaSliderLabel.Text = 'Sigma ';

            % Create SigmaSlider
            app.SigmaSlider = uislider(app.EdgedetectionPanel);
            app.SigmaSlider.Limits = [0.5 10];
            app.SigmaSlider.ValueChangedFcn = createCallbackFcn(app, @SigmaSliderValueChanged, true);
            app.SigmaSlider.Position = [92 163 150 3];
            app.SigmaSlider.Value = 1;

            % Create CornerDetectionPanel
            app.CornerDetectionPanel = uipanel(app.edgeandcornerTab);
            app.CornerDetectionPanel.Title = 'Corner Detection';
            app.CornerDetectionPanel.Position = [376 50 301 255];

            % Create HarrisCornerDetectionButton
            app.HarrisCornerDetectionButton = uibutton(app.CornerDetectionPanel, 'push');
            app.HarrisCornerDetectionButton.ButtonPushedFcn = createCallbackFcn(app, @HarrisCornerDetectionButtonPushed, true);
            app.HarrisCornerDetectionButton.Position = [82 199 140 22];
            app.HarrisCornerDetectionButton.Text = 'Harris Corner Detection';

            % Create MinCornerStrengthSpinnerLabel
            app.MinCornerStrengthSpinnerLabel = uilabel(app.CornerDetectionPanel);
            app.MinCornerStrengthSpinnerLabel.HorizontalAlignment = 'right';
            app.MinCornerStrengthSpinnerLabel.Position = [17 157 113 22];
            app.MinCornerStrengthSpinnerLabel.Text = 'Min Corner Strength';

            % Create MinCornerStrengthSpinner
            app.MinCornerStrengthSpinner = uispinner(app.CornerDetectionPanel);
            app.MinCornerStrengthSpinner.Limits = [0.001 0.1];
            app.MinCornerStrengthSpinner.ValueChangedFcn = createCallbackFcn(app, @MinCornerStrengthSpinnerValueChanged, true);
            app.MinCornerStrengthSpinner.Position = [145 157 100 22];
            app.MinCornerStrengthSpinner.Value = 0.03;

            % Create neighborhoodsizeSpinnerLabel
            app.neighborhoodsizeSpinnerLabel = uilabel(app.CornerDetectionPanel);
            app.neighborhoodsizeSpinnerLabel.HorizontalAlignment = 'right';
            app.neighborhoodsizeSpinnerLabel.Position = [24 110 103 22];
            app.neighborhoodsizeSpinnerLabel.Text = 'neighborhood size';

            % Create neighborhoodsizeSpinner
            app.neighborhoodsizeSpinner = uispinner(app.CornerDetectionPanel);
            app.neighborhoodsizeSpinner.Step = 2;
            app.neighborhoodsizeSpinner.Limits = [3 9];
            app.neighborhoodsizeSpinner.ValueChangedFcn = createCallbackFcn(app, @neighborhoodsizeSpinnerValueChanged, true);
            app.neighborhoodsizeSpinner.Position = [142 110 100 22];
            app.neighborhoodsizeSpinner.Value = 5;

            % Create numberofCornersSpinnerLabel
            app.numberofCornersSpinnerLabel = uilabel(app.CornerDetectionPanel);
            app.numberofCornersSpinnerLabel.HorizontalAlignment = 'right';
            app.numberofCornersSpinnerLabel.Position = [25 61 105 22];
            app.numberofCornersSpinnerLabel.Text = 'number of Corners';

            % Create numberofCornersSpinner
            app.numberofCornersSpinner = uispinner(app.CornerDetectionPanel);
            app.numberofCornersSpinner.Limits = [10 1000];
            app.numberofCornersSpinner.ValueChangedFcn = createCallbackFcn(app, @numberofCornersSpinnerValueChanged, true);
            app.numberofCornersSpinner.Position = [145 61 100 22];
            app.numberofCornersSpinner.Value = 200;

            % Create SelectSpecificRegionButton
            app.SelectSpecificRegionButton = uibutton(app.CornerDetectionPanel, 'push');
            app.SelectSpecificRegionButton.ButtonPushedFcn = createCallbackFcn(app, @SelectSpecificRegionButtonPushed, true);
            app.SelectSpecificRegionButton.Position = [87 17 134 22];
            app.SelectSpecificRegionButton.Text = 'Select Specific Region';

            % Create HOGHoughTab
            app.HOGHoughTab = uitab(app.TabGroup);
            app.HOGHoughTab.Title = 'HOG/Hough';

            % Create UIAxesHist
            app.UIAxesHist = uiaxes(app.HOGHoughTab);
            zlabel(app.UIAxesHist, 'Z')
            app.UIAxesHist.XTick = [];
            app.UIAxesHist.YTick = [];
            app.UIAxesHist.Position = [274 129 230 198];

            % Create DoGLoGPanel
            app.DoGLoGPanel = uipanel(app.HOGHoughTab);
            app.DoGLoGPanel.Title = 'DoG/LoG';
            app.DoGLoGPanel.Position = [12 166 260 152];

            % Create DoGCheckBox
            app.DoGCheckBox = uicheckbox(app.DoGLoGPanel);
            app.DoGCheckBox.ValueChangedFcn = createCallbackFcn(app, @DoGCheckBoxValueChanged, true);
            app.DoGCheckBox.Text = 'DoG';
            app.DoGCheckBox.Position = [13 100 46 22];

            % Create sigma1Label
            app.sigma1Label = uilabel(app.DoGLoGPanel);
            app.sigma1Label.HorizontalAlignment = 'right';
            app.sigma1Label.Position = [6 73 44 22];
            app.sigma1Label.Text = 'sigma1';

            % Create sigma1Spinner
            app.sigma1Spinner = uispinner(app.DoGLoGPanel);
            app.sigma1Spinner.Limits = [1 Inf];
            app.sigma1Spinner.Position = [64 73 56 22];
            app.sigma1Spinner.Value = 15;

            % Create sigma2Label
            app.sigma2Label = uilabel(app.DoGLoGPanel);
            app.sigma2Label.HorizontalAlignment = 'right';
            app.sigma2Label.Position = [131 73 44 22];
            app.sigma2Label.Text = 'sigma2';

            % Create sigma2Spinner
            app.sigma2Spinner = uispinner(app.DoGLoGPanel);
            app.sigma2Spinner.Limits = [2 Inf];
            app.sigma2Spinner.Position = [189 73 56 22];
            app.sigma2Spinner.Value = 20;

            % Create LoGCheckBox
            app.LoGCheckBox = uicheckbox(app.DoGLoGPanel);
            app.LoGCheckBox.ValueChangedFcn = createCallbackFcn(app, @LoGCheckBoxValueChanged, true);
            app.LoGCheckBox.Text = 'LoG';
            app.LoGCheckBox.Position = [17 41 44 22];

            % Create SizeSpinnerLabel
            app.SizeSpinnerLabel = uilabel(app.DoGLoGPanel);
            app.SizeSpinnerLabel.HorizontalAlignment = 'right';
            app.SizeSpinnerLabel.Position = [22 10 28 22];
            app.SizeSpinnerLabel.Text = 'Size';

            % Create SizeSpinner
            app.SizeSpinner = uispinner(app.DoGLoGPanel);
            app.SizeSpinner.Limits = [1 Inf];
            app.SizeSpinner.Position = [65 10 55 22];
            app.SizeSpinner.Value = 15;

            % Create SigmaSpinnerLabel
            app.SigmaSpinnerLabel = uilabel(app.DoGLoGPanel);
            app.SigmaSpinnerLabel.HorizontalAlignment = 'right';
            app.SigmaSpinnerLabel.Position = [136 10 39 22];
            app.SigmaSpinnerLabel.Text = 'Sigma';

            % Create SigmaSpinner
            app.SigmaSpinner = uispinner(app.DoGLoGPanel);
            app.SigmaSpinner.Limits = [1 Inf];
            app.SigmaSpinner.Position = [190 10 53 22];
            app.SigmaSpinner.Value = 1;

            % Create sizeSpinnerLabel
            app.sizeSpinnerLabel = uilabel(app.DoGLoGPanel);
            app.sizeSpinnerLabel.HorizontalAlignment = 'right';
            app.sizeSpinnerLabel.Position = [97 103 26 22];
            app.sizeSpinnerLabel.Text = 'size';

            % Create sizeSpinner
            app.sizeSpinner = uispinner(app.DoGLoGPanel);
            app.sizeSpinner.Limits = [0 Inf];
            app.sizeSpinner.Position = [138 103 62 22];
            app.sizeSpinner.Value = 21;

            % Create HOGPanel
            app.HOGPanel = uipanel(app.HOGHoughTab);
            app.HOGPanel.Title = 'HOG';
            app.HOGPanel.Position = [12 7 260 148];

            % Create HOGCheckBox
            app.HOGCheckBox = uicheckbox(app.HOGPanel);
            app.HOGCheckBox.ValueChangedFcn = createCallbackFcn(app, @HOGCheckBoxValueChanged, true);
            app.HOGCheckBox.Text = 'HOG';
            app.HOGCheckBox.Position = [10 91 49 22];

            % Create BinNumberEditFieldLabel
            app.BinNumberEditFieldLabel = uilabel(app.HOGPanel);
            app.BinNumberEditFieldLabel.HorizontalAlignment = 'right';
            app.BinNumberEditFieldLabel.Position = [76 24 68 22];
            app.BinNumberEditFieldLabel.Text = 'Bin Number';

            % Create BinNumberEditField
            app.BinNumberEditField = uieditfield(app.HOGPanel, 'numeric');
            app.BinNumberEditField.Position = [159 24 55 22];
            app.BinNumberEditField.Value = 9;

            % Create BlockSizeEditFieldLabel
            app.BlockSizeEditFieldLabel = uilabel(app.HOGPanel);
            app.BlockSizeEditFieldLabel.HorizontalAlignment = 'right';
            app.BlockSizeEditFieldLabel.Position = [75 59 61 22];
            app.BlockSizeEditFieldLabel.Text = 'Block Size';

            % Create BlockSizeEditField
            app.BlockSizeEditField = uieditfield(app.HOGPanel, 'numeric');
            app.BlockSizeEditField.Position = [151 59 63 22];
            app.BlockSizeEditField.Value = 2;

            % Create CellSizeEditFieldLabel
            app.CellSizeEditFieldLabel = uilabel(app.HOGPanel);
            app.CellSizeEditFieldLabel.HorizontalAlignment = 'right';
            app.CellSizeEditFieldLabel.Position = [88 91 52 22];
            app.CellSizeEditFieldLabel.Text = 'Cell Size';

            % Create CellSizeEditField
            app.CellSizeEditField = uieditfield(app.HOGPanel, 'numeric');
            app.CellSizeEditField.Position = [155 91 59 22];
            app.CellSizeEditField.Value = 8;

            % Create HoughPanel
            app.HoughPanel = uipanel(app.HOGHoughTab);
            app.HoughPanel.Title = 'Hough';
            app.HoughPanel.Position = [517 1 192 314];

            % Create LineHoughTransformCheckBox
            app.LineHoughTransformCheckBox = uicheckbox(app.HoughPanel);
            app.LineHoughTransformCheckBox.ValueChangedFcn = createCallbackFcn(app, @LineHoughTransformCheckBoxValueChanged, true);
            app.LineHoughTransformCheckBox.Text = 'Line Hough Transform ';
            app.LineHoughTransformCheckBox.Position = [7 259 143 22];

            % Create numPeaksEditFieldLabel
            app.numPeaksEditFieldLabel = uilabel(app.HoughPanel);
            app.numPeaksEditFieldLabel.HorizontalAlignment = 'right';
            app.numPeaksEditFieldLabel.Position = [26 233 65 22];
            app.numPeaksEditFieldLabel.Text = 'num Peaks';

            % Create numPeaksEditField
            app.numPeaksEditField = uieditfield(app.HoughPanel, 'numeric');
            app.numPeaksEditField.Limits = [1 Inf];
            app.numPeaksEditField.Position = [106 233 43 22];
            app.numPeaksEditField.Value = 5;

            % Create CircleHoughTransformCheckBox
            app.CircleHoughTransformCheckBox = uicheckbox(app.HoughPanel);
            app.CircleHoughTransformCheckBox.ValueChangedFcn = createCallbackFcn(app, @CircleHoughTransformCheckBoxValueChanged, true);
            app.CircleHoughTransformCheckBox.Text = 'Circle Hough Transform ';
            app.CircleHoughTransformCheckBox.Position = [14 154 151 22];

            % Create ThresholdSpinnerLabel
            app.ThresholdSpinnerLabel = uilabel(app.HoughPanel);
            app.ThresholdSpinnerLabel.HorizontalAlignment = 'right';
            app.ThresholdSpinnerLabel.Position = [14 68 58 22];
            app.ThresholdSpinnerLabel.Text = 'Threshold';

            % Create ThresholdSpinner
            app.ThresholdSpinner = uispinner(app.HoughPanel);
            app.ThresholdSpinner.Step = 0.1;
            app.ThresholdSpinner.Limits = [0 Inf];
            app.ThresholdSpinner.Position = [87 68 100 22];
            app.ThresholdSpinner.Value = 0.1;

            % Create polarityDropDownLabel
            app.polarityDropDownLabel = uilabel(app.HoughPanel);
            app.polarityDropDownLabel.HorizontalAlignment = 'right';
            app.polarityDropDownLabel.Position = [19 5 44 22];
            app.polarityDropDownLabel.Text = 'polarity';

            % Create polarityDropDown
            app.polarityDropDown = uidropdown(app.HoughPanel);
            app.polarityDropDown.Items = {'bright', 'dark'};
            app.polarityDropDown.Position = [77 5 84 22];
            app.polarityDropDown.Value = 'bright';

            % Create CircleHoughwithpolarityCheckBox
            app.CircleHoughwithpolarityCheckBox = uicheckbox(app.HoughPanel);
            app.CircleHoughwithpolarityCheckBox.ValueChangedFcn = createCallbackFcn(app, @CircleHoughwithpolarityCheckBoxValueChanged, true);
            app.CircleHoughwithpolarityCheckBox.Text = 'Circle Hough with polarity';
            app.CircleHoughwithpolarityCheckBox.Position = [22 35 157 22];

            % Create minRadiusEditFieldLabel
            app.minRadiusEditFieldLabel = uilabel(app.HoughPanel);
            app.minRadiusEditFieldLabel.HorizontalAlignment = 'right';
            app.minRadiusEditFieldLabel.Position = [12 126 62 22];
            app.minRadiusEditFieldLabel.Text = 'minRadius';

            % Create minRadiusEditField
            app.minRadiusEditField = uieditfield(app.HoughPanel, 'numeric');
            app.minRadiusEditField.Limits = [0 Inf];
            app.minRadiusEditField.Position = [89 126 94 22];
            app.minRadiusEditField.Value = 10;

            % Create maxRadiusEditFieldLabel
            app.maxRadiusEditFieldLabel = uilabel(app.HoughPanel);
            app.maxRadiusEditFieldLabel.HorizontalAlignment = 'right';
            app.maxRadiusEditFieldLabel.Position = [8 97 65 22];
            app.maxRadiusEditFieldLabel.Text = 'maxRadius';

            % Create maxRadiusEditField
            app.maxRadiusEditField = uieditfield(app.HoughPanel, 'numeric');
            app.maxRadiusEditField.Position = [88 97 96 22];
            app.maxRadiusEditField.Value = 50;

            % Create fillGapEditFieldLabel
            app.fillGapEditFieldLabel = uilabel(app.HoughPanel);
            app.fillGapEditFieldLabel.HorizontalAlignment = 'right';
            app.fillGapEditFieldLabel.Position = [26 206 39 22];
            app.fillGapEditFieldLabel.Text = 'fillGap';

            % Create fillGapEditField
            app.fillGapEditField = uieditfield(app.HoughPanel, 'numeric');
            app.fillGapEditField.Limits = [0 Inf];
            app.fillGapEditField.Position = [79 206 41 22];
            app.fillGapEditField.Value = 20;

            % Create MinLengthEditFieldLabel
            app.MinLengthEditFieldLabel = uilabel(app.HoughPanel);
            app.MinLengthEditFieldLabel.HorizontalAlignment = 'right';
            app.MinLengthEditFieldLabel.Position = [26 180 61 22];
            app.MinLengthEditFieldLabel.Text = 'MinLength';

            % Create MinLengthEditField
            app.MinLengthEditField = uieditfield(app.HoughPanel, 'numeric');
            app.MinLengthEditField.Limits = [0 Inf];
            app.MinLengthEditField.Position = [101 180 44 22];
            app.MinLengthEditField.Value = 40;

            % Create RANSACTab
            app.RANSACTab = uitab(app.TabGroup);
            app.RANSACTab.Title = 'RANSAC';

            % Create UIAxesransac
            app.UIAxesransac = uiaxes(app.RANSACTab);
            zlabel(app.UIAxesransac, 'Z')
            app.UIAxesransac.XTick = [];
            app.UIAxesransac.YTick = [];
            app.UIAxesransac.Position = [211 108 497 216];

            % Create LoadsecondimageButton
            app.LoadsecondimageButton = uibutton(app.RANSACTab, 'push');
            app.LoadsecondimageButton.ButtonPushedFcn = createCallbackFcn(app, @LoadsecondimageButtonPushed, true);
            app.LoadsecondimageButton.Position = [39 282 119 22];
            app.LoadsecondimageButton.Text = 'Load second image';

            % Create RANSACButton
            app.RANSACButton = uibutton(app.RANSACTab, 'push');
            app.RANSACButton.ButtonPushedFcn = createCallbackFcn(app, @RANSACButtonPushed, true);
            app.RANSACButton.Position = [47 200 100 22];
            app.RANSACButton.Text = 'RANSAC';

            % Create MatchFeaturesDropDown_2Label
            app.MatchFeaturesDropDown_2Label = uilabel(app.RANSACTab);
            app.MatchFeaturesDropDown_2Label.HorizontalAlignment = 'right';
            app.MatchFeaturesDropDown_2Label.Position = [11 241 88 22];
            app.MatchFeaturesDropDown_2Label.Text = 'Match Features';

            % Create MatchFeaturesDropDown_2
            app.MatchFeaturesDropDown_2 = uidropdown(app.RANSACTab);
            app.MatchFeaturesDropDown_2.Items = {'Harris', 'SURF', 'SIFT'};
            app.MatchFeaturesDropDown_2.ValueChangedFcn = createCallbackFcn(app, @MatchFeaturesDropDown_2ValueChanged, true);
            app.MatchFeaturesDropDown_2.Position = [114 241 87 22];
            app.MatchFeaturesDropDown_2.Value = 'SIFT';

            % Create NumberofSamplesSpinnerLabel
            app.NumberofSamplesSpinnerLabel = uilabel(app.RANSACTab);
            app.NumberofSamplesSpinnerLabel.HorizontalAlignment = 'right';
            app.NumberofSamplesSpinnerLabel.Position = [17 170 111 22];
            app.NumberofSamplesSpinnerLabel.Text = 'Number of Samples';

            % Create NumberofSamplesSpinner
            app.NumberofSamplesSpinner = uispinner(app.RANSACTab);
            app.NumberofSamplesSpinner.Limits = [0 Inf];
            app.NumberofSamplesSpinner.Position = [143 170 72 22];
            app.NumberofSamplesSpinner.Value = 1000;

            % Create ConfidenceSpinnerLabel
            app.ConfidenceSpinnerLabel = uilabel(app.RANSACTab);
            app.ConfidenceSpinnerLabel.HorizontalAlignment = 'right';
            app.ConfidenceSpinnerLabel.Position = [22 134 66 22];
            app.ConfidenceSpinnerLabel.Text = 'Confidence';

            % Create ConfidenceSpinner
            app.ConfidenceSpinner = uispinner(app.RANSACTab);
            app.ConfidenceSpinner.Limits = [0 Inf];
            app.ConfidenceSpinner.Position = [103 134 70 22];
            app.ConfidenceSpinner.Value = 99;

            % Create DistancethresholdSpinnerLabel
            app.DistancethresholdSpinnerLabel = uilabel(app.RANSACTab);
            app.DistancethresholdSpinnerLabel.HorizontalAlignment = 'right';
            app.DistancethresholdSpinnerLabel.Position = [23 102 104 22];
            app.DistancethresholdSpinnerLabel.Text = 'Distance threshold';

            % Create DistancethresholdSpinner
            app.DistancethresholdSpinner = uispinner(app.RANSACTab);
            app.DistancethresholdSpinner.Step = 0.1;
            app.DistancethresholdSpinner.Limits = [0 Inf];
            app.DistancethresholdSpinner.Position = [142 102 67 22];
            app.DistancethresholdSpinner.Value = 1.5;

            % Create LinedetectionButton
            app.LinedetectionButton = uibutton(app.RANSACTab, 'push');
            app.LinedetectionButton.ButtonPushedFcn = createCallbackFcn(app, @LinedetectionButtonPushed, true);
            app.LinedetectionButton.Position = [319 81 100 22];
            app.LinedetectionButton.Text = 'Line detection ';

            % Create CircledetectionButton
            app.CircledetectionButton = uibutton(app.RANSACTab, 'push');
            app.CircledetectionButton.ButtonPushedFcn = createCallbackFcn(app, @CircledetectionButtonPushed, true);
            app.CircledetectionButton.Position = [492 81 100 22];
            app.CircledetectionButton.Text = 'Circle detection';

            % Create samplenumberSpinnerLabel
            app.samplenumberSpinnerLabel = uilabel(app.RANSACTab);
            app.samplenumberSpinnerLabel.HorizontalAlignment = 'right';
            app.samplenumberSpinnerLabel.Position = [279 47 88 22];
            app.samplenumberSpinnerLabel.Text = 'sample number';

            % Create samplenumberSpinner
            app.samplenumberSpinner = uispinner(app.RANSACTab);
            app.samplenumberSpinner.Limits = [0 Inf];
            app.samplenumberSpinner.Position = [382 47 57 22];
            app.samplenumberSpinner.Value = 1000;

            % Create distancethresholdSpinnerLabel
            app.distancethresholdSpinnerLabel = uilabel(app.RANSACTab);
            app.distancethresholdSpinnerLabel.HorizontalAlignment = 'right';
            app.distancethresholdSpinnerLabel.Position = [282 16 102 22];
            app.distancethresholdSpinnerLabel.Text = 'distance threshold';

            % Create distancethresholdSpinner
            app.distancethresholdSpinner = uispinner(app.RANSACTab);
            app.distancethresholdSpinner.Limits = [0 Inf];
            app.distancethresholdSpinner.Position = [399 16 66 22];
            app.distancethresholdSpinner.Value = 2;

            % Create StereoVisionTab
            app.StereoVisionTab = uitab(app.TabGroup);
            app.StereoVisionTab.Title = 'Stereo Vision';

            % Create StereoVisionPanel
            app.StereoVisionPanel = uipanel(app.StereoVisionTab);
            app.StereoVisionPanel.Title = 'Stereo Vision';
            app.StereoVisionPanel.Position = [14 16 705 297];

            % Create UIAxesMatches
            app.UIAxesMatches = uiaxes(app.StereoVisionPanel);
            zlabel(app.UIAxesMatches, 'Z')
            app.UIAxesMatches.XTick = [];
            app.UIAxesMatches.YTick = [];
            app.UIAxesMatches.Position = [211 74 488 201];

            % Create LoadSecondImageButton
            app.LoadSecondImageButton = uibutton(app.StereoVisionPanel, 'push');
            app.LoadSecondImageButton.ButtonPushedFcn = createCallbackFcn(app, @LoadSecondImageButtonPushed, true);
            app.LoadSecondImageButton.Position = [20 241 122 22];
            app.LoadSecondImageButton.Text = 'Load Second Image';

            % Create ApplyRANSACButton
            app.ApplyRANSACButton = uibutton(app.StereoVisionPanel, 'push');
            app.ApplyRANSACButton.ButtonPushedFcn = createCallbackFcn(app, @ApplyRANSACButtonPushed, true);
            app.ApplyRANSACButton.Position = [47 159 98 37];
            app.ApplyRANSACButton.Text = 'Apply RANSAC';

            % Create EpipolarlinesIMG1Button
            app.EpipolarlinesIMG1Button = uibutton(app.StereoVisionPanel, 'push');
            app.EpipolarlinesIMG1Button.ButtonPushedFcn = createCallbackFcn(app, @EpipolarlinesIMG1ButtonPushed, true);
            app.EpipolarlinesIMG1Button.Position = [9 107 98 37];
            app.EpipolarlinesIMG1Button.Text = {'Epipolar lines'; ' IMG1'};

            % Create EpipolarlinesIMG2Button
            app.EpipolarlinesIMG2Button = uibutton(app.StereoVisionPanel, 'push');
            app.EpipolarlinesIMG2Button.ButtonPushedFcn = createCallbackFcn(app, @EpipolarlinesIMG2ButtonPushed, true);
            app.EpipolarlinesIMG2Button.Position = [115 106 88 37];
            app.EpipolarlinesIMG2Button.Text = {'Epipolar lines'; ' IMG2'};

            % Create MatchFeaturesDropDownLabel
            app.MatchFeaturesDropDownLabel = uilabel(app.StereoVisionPanel);
            app.MatchFeaturesDropDownLabel.HorizontalAlignment = 'right';
            app.MatchFeaturesDropDownLabel.Position = [11 204 88 22];
            app.MatchFeaturesDropDownLabel.Text = 'Match Features';

            % Create MatchFeaturesDropDown
            app.MatchFeaturesDropDown = uidropdown(app.StereoVisionPanel);
            app.MatchFeaturesDropDown.Items = {'Harris', 'SURF', 'SIFT', 'precomputed'};
            app.MatchFeaturesDropDown.ValueChangedFcn = createCallbackFcn(app, @MatchFeaturesDropDownValueChanged, true);
            app.MatchFeaturesDropDown.Position = [114 204 86 22];
            app.MatchFeaturesDropDown.Value = 'Harris';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = ProgrammingTask3

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end