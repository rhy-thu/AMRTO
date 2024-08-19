clc; clear all; close all;
file_name_field = 'element_pseudo_density_2.csv';
file_name_stl = 'Mesh_2.stl';
file_name_png = 'Mesh';
FIELD = reshape(readmatrix(file_name_field),[30,90,40]); % [nely,nelx,nelz]
CUTOFF = 0.5;                               % The reference pseudo-density
fun_TOSTL(file_name_stl,FIELD,CUTOFF,file_name_png);

function fun_TOSTL(fout,FIELD,CUTOFF,file_name_png)

    [nely, nelx, nelz] = size(FIELD);
    field_extension = zeros(nely+2, nelx+2, nelz+2);
    field_extension(2:end-1,2:end-1,2:end-1) = FIELD;

    %% Smoothing the pseudo-density field
    whether_smooth = 0;
    if whether_smooth ~= 0
        method_smooth = 'box'; % 'box', 'gaussian'  The smoothing method
        size_window = [3,3,3];
        sd_gaussian = 0.5; 
        field_extension = smooth3(field_extension,method_smooth,size_window,sd_gaussian);
    end

    %% Identify the isosurface
    [Xq,Yq,Zq] = meshgrid(0:1:nelx+1, 0:1:nely+1, 0:1:nelz+1);  
    ISO_struct = isosurface(Xq-0.5, Zq-0.5, Yq-0.5, field_extension, CUTOFF);
    faces = ISO_struct.faces;
    vertices = ISO_struct.vertices;
    vertices_single = single(vertices);
    tri_patches = reshape(vertices_single(faces',:)', 3, 3, []);

    %% Calculate the normal vectors
    vector_1 = squeeze(tri_patches(:,2,:) - tri_patches(:,1,:));
    vector_2 = squeeze(tri_patches(:,3,:) - tri_patches(:,1,:));
    normals = cross(vector_1, vector_2);    % The normal vectors of patches
    normals_norm = bsxfun(@times, normals, 1./sqrt(sum(normals.*normals, 1)));
    tri_patches = cat(2, reshape(normals_norm, 3, 1, []), tri_patches);

    %% Draw the isosurface and slice
    whether_plot = 1;
    whether_save_png = 0;
    if whether_plot ~= 0
        w1 = figure(1); clf; set(w1,'Color','w');
        set(w1,'unit','normalized','position',[0.2,0.2,0.5,0.5]);
        set(gca,'position',[0.05,0.05,0.95,0.95] );
        patch(ISO_struct, 'FaceColor', '#0000F0', 'EdgeColor', 'none',...
            'FaceLighting', 'gouraud', 'AmbientStrength', 0.6);
        axis equal tight off; 
        view(45,15); camlight; 
        camproj perspective;

        slice_method = 'linear';  % 'linear', 'cubic', 'nearest'
        w2 = figure(2);
        clf; set(w2,'Color','w');
        set(w2,'unit','normalized','position',[0.2,0.2,0.5,0.5]);
        set (gca,'position',[0.05,0.05,0.95,0.95] );
        xslice = [0 5 10];      % define the cross sections to view
        yslice = ([30 40]);
        zslice = ([0 5 10 15]);
        slice(Xq,Yq,Zq,field_extension, xslice, yslice, zslice, slice_method); % display the slices
        cb = colorbar;                                  % create and label the colorbar
        cb.Label.String = 'Temperature, C';
        axis equal; 
        view(45,15);
        if whether_save_png ~= 0
            exportgraphics(w1,[file_name_png,'_1.png'],'Resolution',600)
            exportgraphics(w2,[file_name_png,'_2.png'],'Resolution',600)
        end
    end

    result_volume = tri_mesh_volume_cal(faces,vertices);
    disp(['Volume = ', num2str(result_volume)]);

    %% Generate the mesh model
    fid = fopen(fout, 'wb+');
    fprintf(fid, '%-80s', 'MESH');                      % Header
    fwrite(fid, size(tri_patches, 3), 'uint32');        % Number of patches
    tri_patches = typecast(tri_patches(:), 'uint16');       
    tri_patches = reshape(tri_patches, 12*2, []);
    tri_patches(end+1, :) = 0;
    fwrite(fid, tri_patches, 'uint16');
    fclose(fid);
    fprintf('%s comprising %d triangular patches was generated.\n', ...
        fout,size(tri_patches, 2));

    whether_cal_curvature = 1;
    if whether_cal_curvature ~= 0
        FV.faces = faces;
        FV.vertices=vertices;
        
        %% calcualte curvatures
        getderivatives = 0;
        [PrincipalCurvatures,PrincipalDir1,PrincipalDir2,FaceCMatrix,VertexCMatrix,Cmagnitude] ...
            = GetCurvatures( FV ,getderivatives);
        GausianCurvature=PrincipalCurvatures(1,:).*PrincipalCurvatures(2,:);

        %% Draw the result 
        w3 = figure(3); clf; set(w3,'Color','w','name','Triangle Mesh Curvature','numbertitle','off');
        set(w3,'unit','normalized','position',[0.2,0.2,0.5,0.5]);
        set(gca,'position',[0.05,0.05,0.95,0.95] );
        colormap cool
        clim([min(GausianCurvature) max(GausianCurvature)]); % color overlay the gaussian curvature
        mesh_h=patch(FV,'FaceVertexCdata',GausianCurvature','facecolor','interp','edgecolor',...
            'interp','EdgeAlpha',0.2);
        %set some visualization properties
        set(mesh_h,'ambientstrength',0.5);
        axis equal; 
        view(45,15);
        camlight();
        lighting phong
        colorbar();
    end

end

function result_volume = tri_mesh_volume_cal(Tri,V)
if size(Tri,2)~=3
    error('This function is intended ONLY for triangular surface meshes')
end
% Face vertices
V1=V(Tri(:,1),:);
V2=V(Tri(:,2),:);
V3=V(Tri(:,3),:);
% Face centroids
C=(V1+V2+V3)/3;
% Face normals
FN=cross(V2-V1,V3-V1,2);
% Volume
result_volume=sum(dot(C,FN,2))/6;
end

% The program was developed based on the code in the following paper:
% Liu, K., Tovar, A. An efficient 3D topology optimization code 
% written in Matlab. Struct Multidisc Optim 50, 1175–1196 (2014). 
% https://doi.org/10.1007/s00158-014-1107-x