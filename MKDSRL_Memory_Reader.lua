-- Lua Script for checkpoint color-marking and speed marker
-- Purpose is to give PyCharm program good indicator for if the bot is going the wrong direction
-- And if bot is driving at a good speed
----- SCRIPT OPTIONS -----
local verUSA = true -- set to false if using a EUR ROM
local exactSine = false -- set to true to read sine/cosine as int (no rounding means no rounding errors)
local playerID = 0 -- set to 1 to watch your ghost's values
--------------------------
 
local pntCheckNum -- Checkpoint number
local pntPlayerData -- X, Y, Z, speed
local pntCheckData -- Checkpoint data
 
local checkpoint = 0
local xPos, zPos, xPosPrev, zPosPrev, realspeed = 0,0,0,0 -- Positions to determine speed
local angle, driftangle, ref_angle = 0,0,0 -- Angle positions to help determine wrong direction
local chkpnt_ang = {16380, 16380, 20805, 24812, 28650, -30861, -23773, -17975, -15386, -11925,
                    -8233, -8233, -8233, -8233, -8233, -8233, -14276, -18273, -27352, 29514, 26495, 
                    18165, 16380, 16380, 16380, 16380} -- Angles corresponding to right direction at each checkpoint

local b_val = 0 -- Value of blue value that will be displayed
local g_val = 0 -- Value of green value that will be displayed

function is_over_90(angle, ref_angle) -- Wrong direction determination
  local reangle = 0
  local reref_angle = 0
  if angle < 0 then reangle = 65520 + angle else reangle = angle end
  if ref_angle < 0 then reref_angle = 65520 + ref_angle else reref_angle = ref_angle end
  if math.min((math.abs(reref_angle - reangle) % 65520), (65520 - math.abs(reref_angle - reangle) % 65520)) > 16380 then
    return true
  else
    return false
  end
end

function fn()
    -- Read pointer values
    pntCheckNum = memory.readdword(0x021661B0)
    pntCheckData = memory.readdword(0x02175600)
    if (verUSA) then
        pntPlayerData = memory.readdword(0x217ACF8)
    else
        pntPlayerData = memory.readdword(0x217D028)
    end
    pntPlayerData = playerID * 0x5A8 + pntPlayerData
 
    -- Read checkpoint value
    checkpoint = memory.readbytesigned(pntCheckNum + 0xDAE)
    -- Read angle vals
    pAng = angle
    pDAng = driftAngle
    angle = memory.readwordsigned(pntPlayerData + 0x236)
    driftAngle = memory.readwordsigned(pntPlayerData + 0x388)
    -- Read position values to get speed
    xPosPrev = xPos
    zPosPrev = zPos
    xPos = memory.readdwordsigned(pntPlayerData + 0x80)
    zPos = memory.readdwordsigned(pntPlayerData + 0x80 + 8)
    realspeed = math.sqrt(math.abs(zPosPrev - zPos) * math.abs(zPosPrev - zPos) + math.abs(xPosPrev - xPos) * math.abs(xPosPrev - xPos))
    -- Get color values from speed and checkpoint
    if type(chkpnt_ang[checkpoint + 1]) == 'nil' then ref_angle = 0 else ref_angle = chkpnt_ang[checkpoint + 1] end
    if is_over_90(angle, ref_angle) then b_val = 200 else b_val = 100 end 
    if (realspeed > 40000.0) then g_val = 200 else if (realspeed > 20000) then g_val = 150 else if (realspeed > 10000) then g_val = 100 else g_val = 50 end end end
    
end
 
function fm() -- Image display
    -- Display checkpoints
    if (checkpoint > -1) then gui.box(0, 0, 10, 10, {r = 80, g = g_val, b = b_val, a = 255}) end -- Error correction for if checkpoint is negative (between races mostly)
end
 
emu.registerafter(fn) -- Performing two functions
gui.register(fm)