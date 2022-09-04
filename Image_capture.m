for i=1:      % i range should be the same as the patch number
    for j=1:15
        if j == 14
        else
            image_collect=Patches{i,1}(:,:,j);
            image = mat2gray(image_collect);
            name=num2str(j)+"_of_"+num2str(i);
            save(name+'.mat','image')
        end
    end
end
