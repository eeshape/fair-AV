# import carla
# import sys

# try:
#     # CARLA 서버에 연결
#     client = carla.Client('127.0.0.1', 2000)
#     client.set_timeout(10.0)
    
#     # 월드 가져오기
#     world = client.get_world()
    
#     print(f"✓ CARLA 연결 성공!")
#     print(f"  - 맵: {world.get_map().name}")
#     print(f"  - CARLA 버전: {client.get_server_version()}")
    
#     # 사용 가능한 블루프린트 확인
#     blueprint_library = world.get_blueprint_library()
    
#     # 차량 블루프린트 확인
#     vehicles = blueprint_library.filter('vehicle.*')
#     print(f"  - 사용 가능한 차량: {len(vehicles)}개")
    
#     # 보행자 블루프린트 확인
#     walkers = blueprint_library.filter('walker.*')
#     print(f"  - 사용 가능한 보행자: {len(walkers)}개")
    
#     # 구체적인 보행자 필터 테스트
#     pedestrians = blueprint_library.filter('walker.pedestrian.*')
#     print(f"  - walker.pedestrian.* 필터: {len(pedestrians)}개")
    
#     if len(pedestrians) == 0:
#         print("\n⚠ walker.pedestrian.* 필터로 보행자를 찾을 수 없습니다.")
#         print("  사용 가능한 보행자 블루프린트:")
#         for walker in walkers[:10]:  # 처음 10개만 출력
#             print(f"    - {walker.id}")
    
#     # 스폰 포인트 확인
#     spawn_points = world.get_map().get_spawn_points()
#     print(f"  - 스폰 포인트: {len(spawn_points)}개")
    
# except Exception as e:
#     print(f"✗ CARLA 연결 실패!")
#     print(f"  에러: {e}")
#     print("\n해결 방법:")
#     print("  1. CARLA 서버가 실행 중인지 확인")
#     print("  2. 포트 번호가 올바른지 확인 (기본: 2000)")
#     print("  3. 방화벽 설정 확인")
#     sys.exit(1)



from carla import ActorAttributeType

def attr_value(attr_obj):
    t = attr_obj.type
    if t == ActorAttributeType.Bool:
        return attr_obj.as_bool()
    if t == ActorAttributeType.Int:
        return attr_obj.as_int()
    if t == ActorAttributeType.Float:
        return attr_obj.as_float()
    if t == ActorAttributeType.RGBColor:
        c = attr_obj.as_color()
        return (c.r, c.g, c.b)
    return attr_obj.as_str()

for bp in blueprints:
    info = {"id": bp.id}
    for attr in ["gender", "age", "color", "speed", "can_use_wheelchair"]:
        if bp.has_attribute(attr):
            info[attr] = attr_value(bp.get_attribute(attr))
    rows.append(info)
