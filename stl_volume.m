clc;
if 1
    file_name = 'data/MyTop3d1205.stl';
    mesh=stlread(file_name);
    Tri_mesh = mesh.ConnectivityList;
    Ver_mesh = mesh.Points;
else   
    objData = readObj('data/MyTop3d1205_obj.obj');
    Ver_mesh = objData.v;
    Tri_mesh = objData.f.v;
end


vol = tri_mesh_volume_cal(Tri_mesh,Ver_mesh);
fprintf('%.2f\n', vol); 


function result_volume = tri_mesh_volume_cal(Tri,V)
if size(Tri,2) == 4
    Tri = [Tri(:,[1,2,3]);Tri(:,[1,3,4])];
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

