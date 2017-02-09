###常用命令

#### 关联 connect to github with SSH  
<code> https://help.github.com/articles/connecting-to-github-with-ssh/</code>  
#### 项目初始，从远端克隆到本地  
<code>git clone https://github.com/Entheos1994/adml_dataminer.git(adml_dataminer 远程库)</code>  
#### 从远端pull到本地，同一目录下   
<code> git branch #检查当前branch</code>  
<code> git status #查看当前git状态</code>  
<code> git pull origin master #若远端领先本地版本,更新本地为新版本</code>  

#### 从本地上传到远端(在远端合并)  
<code> git branch branchname(新建分支) </code>  
<code> git checkout branchname #切换到新分支 </code>  
.........................................在本地分支上对代码操作更新<br>  
<code> git status #检查状态</code>  
<code> git add .  #添加更新</code>  
<code> git commit -m '本次commit的命名'</code>  
<code> git push origin branchname #push到远端</code>  

####其他  
<code> git checkout -d branchname #删除分支</code> 
<code> 如果git pull 显示'no tracking information'</code>  
<code> git branch --set-upstream branchname origin/branchname</code>  
<code> git remote -v #查看远程库信息</code>  
<code> git log --pretty=oneline</code>  
<code> git reset --hard commit_id # 回退到某个commit</code>  
<code> git checkout -- 文件名 #把commit的文件撤回</code>  
<code> git diff 文件名 #查看修改前后的不同</code>  

####一般流程 (han为分支名称)  
<code> git pull origin master</code>  
<code> git branch han</code>  
......do.....some.......something...  
<code> git add .</code>  
<code> git commit -m 'first update'</code>  
<code> git push origin han</code>  
<code> 等待分支合并</code>  
<code> git branch -d han #删除分支</code>  

####source  
- http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000 廖雪峰git教程  
- https://git-scm.com/doc git documentation  