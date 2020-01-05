### 前言

这不是第一次部署博客，前年（对..都前年了）折腾过几天，后来由于太懒就闲置了

等后来再想重新开始的时候，发现之前用的两台电脑都还给学校了..源文件当时也不知道要保存，然后又陷入了源文件不在想恢复的过程...然而使用`hexo d`是经过转换后部署到`github`的，并不是源文件。所以，又拖拖拖（sad

### Hexo博客在多台电脑提交和更新

- 结论：是可以的，分别托管page和source即可。因为部署好的github.io是经过转换的，直接clone下来并不是"原先的文件"

网络有很多方法，大同小异。比较推崇的是在github.io的项目中新建一个分支用于存放源文件。

1. 新建分支`hexo`(可自己命名)，将这个分支设为default

2. 使用三部曲提交即可（就是把它当成一个文件项目）

   ```shell
   git add . 
   git commit -m "" 
   git push -u origin hexo:hexo #以后每次提交可以直接git push
   ```

3. **无需切换到master分支**，直接在`hexo`分支用三部曲提交部署，会自动更新在master。

   ```powershell
   hexo clean
   hexo g
   hexo s #（可在本地先查看) 
   hexo d
   ```

4. 注意：git多分支，当切换分支时，本地的文件会相应变化，自动“变成”工作分支的最后一次提交文件。所以你此时的所有工作都在`hexo`分支上，就不要切回master了！之前还没用过分支功能，以为部署要切回去，然后切回去什么文件都没了...吓死

5. 去一台新电脑的时候

   - 首次`clone hexo`分支的文件，安装`hexo`相关插件和软件。包括`node.js,git,npm`相关。（由于`hexo`的这些依赖可以直接安装，所以提交的时候仓库的`.gitignore`文件自动忽略了这些，这里需要重新安装，很方便的）

     ```shell
     # 安装git，配置好
     # 安装node.js
     node -v #查看是否安装好
     # 安装hexo
     npm install hexo-cli -g
     # 初始化博客目录，无须先建文件夹
     hexo init name.github.io
     # 进入目录：
     cd name.github.io
     # 安装npm
     npm install
     # 可以生成一个全新的博客
     hexo clean
     hexo g
     hexo s
     # 在本地http://localhost:4000可以看见一个全新的blog，如同你之前新建的时候一样
     # 关联远程仓
     git remote add origin https://github.com/yourname/yourname.github.io.git 
     # 我是在github上直接建的分支，远程仓有而本地仓没有
     # 本地新建分支hexo，并自动跟踪远程的同名分支
     git checkout --track origin/hexo
     ```

   - 除了首次去一台**全新的未布置过hexo博客的**电脑以外。

     ```shell
     git pull # 恢复上一次在任一电脑上push上去的文件 git pull origin hexo:hexo
     # 一通操作,记得
     git add
     git commit
     git push 
     # 再
     # 删除io文件夹下的 .deploy_git 文件夹
     # 最后
     hexo clean
     hexo g
     hexo d 
     # 即可
     # 如果产生冲突，实在不行也可暴力解决，反正也是要更新到上一次的文件版本
     git fetch origin hexo
     ```

### 遇到的问题及解决

- hexo d后出现 ERROR Deployer not found: git

 这是因为没安装`hexo-deployer-git`插件，在**站点目录**下输入下面的插件安装就好了 

```shell
 npm install --save hexo-deployer-git 
```

- error: RPC failed; curl 56 OpenSSL SSL_read: SSL_ERROR_SYSCALL, errno 10054；fatal: The remote end hung up unexpectedly

提交的文件太大，导致推送失败-->全局设置大一点的文件大小

```shell
git config http.postBuffer 524288000
#查看是否设置成功
git config --list
```

- git branch 看不到任何信息

```shell
# 先commit再查看分支branch！！
git brach
git checkout BranchName # 切换分支
git checkout -b BranchName # 创建分支BanchName并切换
```

- 提交的时候新主题下的文件没有push上去

删除theme/chic文件夹下的.git文件即可。git无法多层提交

- 修改了一些东西，在本地`hexo s`是ok的，但`hexo d`部署之后没有显示修改

删除name.github.io文件夹下的 `.deploy_git` 文件夹，再次`hexo`三步曲

### 参考文献

[hexo多电脑同步管理一个博客 | ZYMIN]( https://www.jianshu.com/p/842897f0c8ba )

[利用Hexo在多台电脑上提交和更新github pages博客]( https://www.jianshu.com/p/0b1fccce74e0 )

[git 关联远程分支](https://www.cnblogs.com/ampl/p/10873877.html)



