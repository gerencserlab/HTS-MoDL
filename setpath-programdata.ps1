[Environment]::SetEnvironmentVariable(
  "PATH",
  "C:\ProgramData\Image Analyst\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;" + [Environment]::GetEnvironmentVariable("PATH", "User"),
  "User"
)
