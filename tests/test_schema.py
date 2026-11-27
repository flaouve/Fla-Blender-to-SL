import json
from fla_blender_to_sl.schema import (
    BlockType,
    LightType,
    PrimitiveFlags,
    PrimitiveType,
    SchematicObjectDataList,
    ShadowType,
    Vec2,
    Vec3,
    make_light_block,
    make_primitive_block,
    make_text_block,
)


def test_primitive_block_matches_fla_sl_schema():
    block = make_primitive_block(
        name="Wall_01",
        object_id=1001,
        parent_id=1000,
        position=Vec3(1.0, 2.0, 3.0),
        rotation=Vec3(0.0, 90.0, 0.0),
        scale=Vec3(4.0, 3.0, 0.2),
        primitive_type=PrimitiveType.CUBE,
        color="434B56FF",
        flags=PrimitiveFlags.COLLIDABLE | PrimitiveFlags.VISIBLE,
        static=True,
    )
    doc = SchematicObjectDataList(root_object_id=1000, blocks=[block])
    parsed = json.loads(doc.to_json())

    assert parsed["RootObjectId"] == 1000
    b = parsed["Blocks"][0]
    assert b["Name"] == "Wall_01"
    assert b["ObjectId"] == 1001
    assert b["ParentId"] == 1000
    assert b["BlockType"] == 1
    assert "AnimatorName" not in b
    assert b["Properties"] == {
        "PrimitiveType": 3,
        "Color": "434B56FF",
        "PrimitiveFlags": 3,
        "Static": True,
    }


def test_light_block_scales_intensity_and_range_by_100():
    block = make_light_block(
        name="CeilingLight",
        object_id=1002,
        parent_id=1000,
        position=Vec3(0.0, 3.0, 0.0),
        rotation=Vec3(90.0, 0.0, 0.0),
        light_type=LightType.POINT,
        color="FFFFFFFF",
        unity_intensity=1.5,
        unity_range=10.0,
        shadow_type=ShadowType.SOFT,
    )
    props = block.to_dict()["Properties"]
    assert props["LightType"] == 2
    assert props["Intensity"] == 150.0
    assert props["Range"] == 1000.0
    assert props["Shape"] == 0
    assert props["SpotAngle"] == 30.0
    assert props["InnerSpotAngle"] == 21.8
    assert props["ShadowStrength"] == 1.0
    assert props["ShadowType"] == 2


def test_text_block_formats_tmp_tags_and_omits_z_in_display_size():
    block = make_text_block(
        name="Sign",
        object_id=1003,
        parent_id=1000,
        position=Vec3(0.0, 2.0, 0.0),
        rotation=Vec3(0.0, 180.0, 0.0),
        scale=Vec3(0.03, 0.03, 0.03),
        text="SITE-83",
        color="FFFFFFFF",
        font_size=14.0,
        align="center",
        display_size=Vec2(20.0, 5.0),
    )
    props = block.to_dict()["Properties"]
    assert props["Text"] == "<align=center><color=#FFFFFFFF><size=14.0>SITE-83</size></color></align>"
    assert props["DisplaySize"] == {"x": 20.0, "y": 5.0}
    assert "z" not in props["DisplaySize"]
