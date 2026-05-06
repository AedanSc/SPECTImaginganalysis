robotModel = loadrobot("kukaIiwa14");
robotModel.DataFormat = 'column';
cameraToEndEffector = load("cameraToEndEffectorTform.mat").cameraToEndEffectorTform;
cameraToEndEffectorTform = cameraToEndEffector.A;
endEffectorCamera = invert(cameraToEndEffector);
endEffectorCameraTform = endEffectorCamera.A;
endEffectorToBaseTform = getTransform(robotModel, [-12, -40, 10, -92, 3, 95, 0]', "iiwa_link_ee_kuka");
cameraToBaseTform = endEffectorToBaseTform * cameraToEndEffectorTform;

cameraToBaseTform %% THIS IS THE TRANSFORMATION MATRIX