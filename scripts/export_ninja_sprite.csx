// Exports Ninpek's player run frames and origin for scripts/render_ghosts.py.
// Usage: dotnet <UndertaleModCli.dll> load "<game dir>/data.win" -s scripts/export_ninja_sprite.csx < /dev/null
// Writes to $SPRITE_DIR, default runs/ninpek/ghosts/sprite.
using UndertaleModLib.Util;

EnsureDataLoaded();
string dir = Environment.GetEnvironmentVariable("SPRITE_DIR") ?? "runs/ninpek/ghosts/sprite";
Directory.CreateDirectory(dir);
var spr = Data.Sprites.ByName("s34_NinjaRun");
using (var worker = new TextureWorker())
{
    for (int i = 0; i < spr.Textures.Count; i++)
    {
        worker.ExportAsPNG(spr.Textures[i].Texture, Path.Combine(dir, $"run_{i}.png"), null, true);
    }
}
File.WriteAllText(Path.Combine(dir, "origin.txt"), $"{spr.OriginX} {spr.OriginY} {spr.Width} {spr.Height}\n");
