// UndertaleModCli script: registers ufo50rl.dll and adds the lockstep sync point to oController.
// Shared buffer layout (must match src/platform/lockstep.rs):
//   [0, 8)     keys to hold (VK codes, 0 = empty), written by the trainer
//   [8, 12)    u32 frames to run before the next sync (min 1)
//   [12, 16)   u32 flags: 1 = quit, 2 = launch game (id in bits 8..15, Ninpek start camera x / 16 in bits 16..31)
//   [16, 144)  16 x f64 game state: frame, room, game id, paused, then Ninpek's
//              o34_Mas points[0], life[0], state, iWater, then title menuSel,
//              numPlayers, demo playback, camera x, run kill count, boss HP (see GameState in lockstep.rs)
//   [144, ..)  384x216 BGRA frame (native resolution; the trainer downscales it)
using UndertaleModLib.Compiler;
using UndertaleModLib.Models;

UndertaleString S(string s) => Data.Strings.MakeString(s);

uint nextId = 0;
foreach (var e in Data.Extensions)
    foreach (var f in e.Files)
        foreach (var fn in f.Functions)
            nextId = Math.Max(nextId, fn.ID);

UndertaleExtensionFunction Fn(string name, params UndertaleExtensionVarType[] args)
{
    var fn = new UndertaleExtensionFunction { Name = S(name), ExtName = S(name), ID = ++nextId, Kind = 11, RetType = UndertaleExtensionVarType.Double };
    foreach (var a in args)
        fn.Arguments.Add(new UndertaleExtensionFunctionArg { Type = a });
    Data.Functions.EnsureDefined(name, Data.Strings);
    return fn;
}

// Mirrors how the game's Steamworks DLL is registered.
var file = new UndertaleExtensionFile { Filename = S("ufo50rl.dll"), InitScript = S(""), CleanupScript = S(""), Kind = Data.Extensions[0].Files[0].Kind };
file.Functions.Add(Fn("rl_init"));
file.Functions.Add(Fn("rl_step", UndertaleExtensionVarType.String, UndertaleExtensionVarType.Double));
var ext = new UndertaleExtension { Name = S("ufo50rl"), ClassName = S(""), Version = S("1.0.0"), FolderName = S("") };
ext.Files.Add(file);
Data.Extensions.Add(ext);
Data.FORM.EXTN.productIdData.Add(new byte[16]);
Data.GlobalFunctions = null;

var g = new CodeImportGroup(Data);

g.QueueAppend("gml_Object_oController_Create_0", @"
global.rlOn = rl_init();
if (global.rlOn)
{
    global.rlBuf = buffer_create(331920, buffer_fixed, 1);
    global.rlSurf = -1;
    global.rlKeys = array_create(8, 0);
    global.rlPrevKeys = array_create(8, 0);
    global.rlWait = 0;
    global.rlFrame = 0;
    global.rlStartX = 0;
    audio_master_gain(0);
}
");

g.QueueReplace("gml_Object_oController_Draw_77", @"
if (!global.rlOn)
{
    exit;
}
game_set_speed(10000, gamespeed_fps);
// Keeps training runs out of the player's save file, high scores and achievements.
global.cheatedNoSave = true;
global.rlPrevKeys = global.rlKeys;
// An idle title screen starts a recorded demo that ignores input.
with (oTitleScreens)
{
    attractTimer = 0;
}
global.rlFrame++;
// scrInitDisplay turns vsync on at startup, and the default timing method also waits on vblanks.
if (global.rlFrame == 2)
{
    display_reset(0, false);
    display_set_timing_method(tm_sleep);
    display_set_sleep_margin(0);
}
// Ninpek: move a fresh run to rlStartX, mirroring the game's own run start (o34_Mas state 10)
// and death respawn, so spawns, coins and the water section stay consistent with the camera.
if (global.rlStartX > 0 && instance_exists(o34_Mas) && o34_Mas.state == 20 && !variable_instance_exists(o34_Mas, ""rlWarped""))
{
    var _x = global.rlStartX;
    global.rlStartX = 0;
    with (o34_Mas)
    {
        rlWarped = true;
        set_xview(_x);
        with (o34_eSpawn)
        {
            if (x < _x + 400)
            {
                // Starting past the level-end marker still has to bring in the boss.
                if (x >= _x || spawn == o34_BossMas)
                {
                    instance_create(x, y, spawn);
                }
                instance_destroy();
            }
        }
        with (oEnemy)
        {
            if (x < _x)
            {
                instance_destroy();
            }
        }
        with (o34_BonusSmall)
        {
            if (x < _x)
            {
                instance_destroy();
            }
            else if (x < _x + 400)
            {
                visible = true;
            }
        }
        iWater = (_x > 11328) ? 3 : ((_x > 7488) ? 2 : ((_x > 3648) ? 1 : 0));
        with (o34_Water)
        {
            iWater = other.iWater;
        }
        if (verify(p[0]))
        {
            var _ghost = instance_create(_x + 192, camera_get_view_y(view_get_camera(0)) - 48, o34_Ghost);
            _ghost.sprite_index = s34_NinjaGhost;
            _ghost.num = 0;
            _ghost.myWpn = p[0].myWpn;
            with (p[0])
            {
                instance_destroy();
            }
        }
    }
}
if (global.rlWait > 0)
{
    global.rlWait--;
    exit;
}
var _b = global.rlBuf;
buffer_poke(_b, 16, buffer_f64, global.rlFrame);
buffer_poke(_b, 24, buffer_f64, room);
buffer_poke(_b, 32, buffer_f64, global.currGameID);
buffer_poke(_b, 40, buffer_f64, global.paused);
// o34_Mas exists on the title screen before its run arrays are set.
var _mas = instance_exists(o34_Mas);
var _run = _mas && variable_instance_exists(o34_Mas, ""points"") && variable_instance_exists(o34_Mas, ""life"");
buffer_poke(_b, 48, buffer_f64, _run ? o34_Mas.points[0] : -1);
buffer_poke(_b, 56, buffer_f64, _run ? o34_Mas.life[0] : -99);
buffer_poke(_b, 64, buffer_f64, (_mas && variable_instance_exists(o34_Mas, ""state"")) ? o34_Mas.state : -1);
buffer_poke(_b, 72, buffer_f64, (_run && variable_instance_exists(o34_Mas, ""iWater"")) ? o34_Mas.iWater : -1);
var _title = instance_exists(oTitleScreens) && variable_instance_exists(oTitleScreens, ""menuSel"");
buffer_poke(_b, 80, buffer_f64, _title ? oTitleScreens.menuSel : -1);
buffer_poke(_b, 88, buffer_f64, global.numPlayers);
buffer_poke(_b, 96, buffer_f64, inRecordOrPlayback());
buffer_poke(_b, 104, buffer_f64, _run ? camera_get_view_x(view_get_camera(0)) : -1);
// scr34_Init zeroes this when a run starts; scr34_eDamage counts every enemy kill that isn't a mine, shroom or boss.
buffer_poke(_b, 112, buffer_f64, (_run && variable_global_exists(""g34_totalKills"")) ? global.g34_totalKills : -1);
// The level-end eye (35 HP) is Ninpek's only boss.
buffer_poke(_b, 120, buffer_f64, (_run && instance_exists(o34_BossEye)) ? o34_BossEye.life : -1);
if (!surface_exists(global.rlSurf))
{
    global.rlSurf = surface_create(384, 216);
}
var _filter = gpu_get_texfilter();
gpu_set_texfilter(false);
surface_set_target(global.rlSurf);
draw_clear(c_black);
draw_surface_stretched(application_surface, 0, 0, 384, 216);
surface_reset_target();
gpu_set_texfilter(_filter);
buffer_get_surface(_b, global.rlSurf, 144);
if (!rl_step(buffer_get_address(_b), 331920))
{
    game_end();
    exit;
}
var _keys = array_create(8, 0);
for (var i = 0; i < 8; i++)
{
    _keys[i] = buffer_peek(_b, i, buffer_u8);
}
global.rlKeys = _keys;
global.rlWait = max(1, buffer_peek(_b, 8, buffer_u32)) - 1;
var _flags = buffer_peek(_b, 12, buffer_u32);
if (_flags & 1)
{
    game_end();
}
if (_flags & 2)
{
    var _id = (_flags >> 8) & 255;
    global.rlStartX = ((_flags >> 16) & 65535) * 16;
    var _slot = 0;
    for (var i = 1; i <= global.NUM_GAMES; i++)
    {
        if (global.mGameID[i] == _id)
        {
            _slot = i;
        }
    }
    if (global.paused)
    {
        scrUnpause();
    }
    global.attractMode = false;
    global.playbackMode = false;
    // Same sequence as the in-game launchers (o51_LX2): the title screen waits until this data is loaded.
    load_datagroup(_id, true, true);
    scrPlayGame(_slot);
}
");

// GameMaker drops keyboard state for unfocused windows, so with several game windows only one
// would see simulated key presses. Feed the trainer's keys straight into the game's input script.
// Keys are matched against the default bindings (arrows, X = fire1, Z = fire2, D/Esc = start).
// The script holds `function scrGetInput(arg0)`, so the code goes inside the body, ahead of its first line.
g.QueueFindReplace("gml_GlobalScript_scrGetInput", "var key_map = global.keyMap[arg0];", @"
if (variable_global_exists(""rlOn"") && global.rlOn)
{
    var _k = (arg0 == 0) ? global.rlKeys : [];
    var _p = (arg0 == 0) ? global.rlPrevKeys : [];
    holdUp = array_contains(_k, vk_up);
    holdDown = array_contains(_k, vk_down);
    holdLeft = array_contains(_k, vk_left);
    holdRight = array_contains(_k, vk_right);
    pressUp = holdUp && !array_contains(_p, vk_up);
    pressDown = holdDown && !array_contains(_p, vk_down);
    pressLeft = holdLeft && !array_contains(_p, vk_left);
    pressRight = holdRight && !array_contains(_p, vk_right);
    fire1 = array_contains(_k, 88);
    fire2 = array_contains(_k, 90);
    fire1pressed = fire1 && !array_contains(_p, 88);
    fire2pressed = fire2 && !array_contains(_p, 90);
    fire1released = !fire1 && array_contains(_p, 88);
    fire2released = !fire2 && array_contains(_p, 90);
    var _start = array_contains(_k, 68) || array_contains(_k, vk_escape);
    pressStart = _start && !(array_contains(_p, 68) || array_contains(_p, vk_escape));
    if (holdLeft && holdRight)
    {
        holdLeft = false;
        holdRight = false;
        pressLeft = false;
        pressRight = false;
    }
    if (holdUp && holdDown)
    {
        holdUp = false;
        holdDown = false;
        pressUp = false;
        pressDown = false;
    }
    exit;
}
var key_map = global.keyMap[arg0];");

g.Import();
