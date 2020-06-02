SELECT userr.first_name, userr.last_name, node.name, node.cpu, node.gpu
	FROM public.laba_user userr, public.user_node usernode, public.node node where userr.id = usernode.user_id and usernode.node_id = node.id; 118 ms 112 ms 106 ms

SELECT file_descriptor.id, video.id, camera.id, userr.id, node.id FROM public.file_descriptor, public.video,
public.camera, public.laba_user userr, public.user_node usernode, public.node WHERE file_descriptor.id = video.file_descriptor_id
AND video.camera_id = camera.id AND camera.user_id = userr.id AND userr.id = usernode.user_id AND usernode.node_id = node.id; 

SELECT * FROM public.video; 1509 ms 1777 ms 1375 ms

SELECT * FROM public.file_descriptor, public.video, public.camera, public.laba_user userr WHERE file_descriptor.id = video.file_descriptor_id 
AND video.camera_id = camera.id AND camera.user_id = userr.id AND file_descriptor.size < 0.5; 5158 ms 5424 ms 4203 ms

SELECT userr.id, COUNT(usernode.node_id) FROM public.laba_user userr, public.user_node usernode WHERE userr.id = usernode.user_id GROUP BY userr.id HAVING COUNT(usernode.node_id) > 1; 138 ms 198 ms 102 ms
